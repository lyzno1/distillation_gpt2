import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
import logging
import os
from tqdm import tqdm
import json

class DistillationTrainer:
    def __init__(self, config, student_model, train_dataset, val_dataset=None):
        self.config = config
        self.student = student_model
        self.teacher = GPT2LMHeadModel.from_pretrained(config.teacher_model_type)
        
        # 冻结教师模型参数
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # 将模型移至GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        self.teacher.to(self.device)
        
        # 设置为评估模式
        self.teacher.eval()
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=4
            )
        else:
            self.val_loader = None
            
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def compute_loss(self, student_logits, teacher_logits, labels):
        """计算蒸馏损失和任务损失"""
        # 蒸馏损失 - KL散度
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.config.temperature, dim=-1),
            F.softmax(teacher_logits / self.config.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        # 任务损失 - 交叉熵
        task_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),  # (batch * seq_len, vocab_size)
            labels.view(-1)                                     # (batch * seq_len)
        )
        
        # 总损失
        loss = self.config.alpha * distillation_loss + (1 - self.config.alpha) * task_loss
        
        return loss, distillation_loss, task_loss
    
    def train_step(self, batch):
        """单个训练步骤"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # 获取教师模型的预测
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids)
            teacher_logits = teacher_outputs.logits
            
        # 获取学生模型的预测
        student_outputs = self.student(input_ids)
        student_logits = student_outputs[0]
        
        # 计算损失
        loss, distill_loss, task_loss = self.compute_loss(
            student_logits, teacher_logits, labels
        )
        
        # 反向传播
        loss.backward()
        
        return loss.item(), distill_loss.item(), task_loss.item()
    
    def train(self):
        """训练循环"""
        self.logger.info("开始训练...")
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.student.train()
            total_loss = 0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch+1}") as pbar:
                for step, batch in enumerate(pbar):
                    # 梯度累积
                    if step % self.config.gradient_accumulation_steps == 0:
                        self.optimizer.zero_grad()
                        
                    loss, distill_loss, task_loss = self.train_step(batch)
                    
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(), 1.0
                        )
                        self.optimizer.step()
                        
                    total_loss += loss
                    global_step += 1
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'distill_loss': f"{distill_loss:.4f}",
                        'task_loss': f"{task_loss:.4f}"
                    })
                    
                    # 验证
                    if global_step % self.config.eval_steps == 0:
                        val_loss = self.evaluate()
                        self.logger.info(f"Step {global_step}: Validation Loss: {val_loss:.4f}")
                        
                        # 保存最佳模型
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            self.save_model("best_model")
                            
                    # 定期保存检查点
                    if global_step % self.config.save_steps == 0:
                        self.save_model(f"checkpoint-{global_step}")
                        
            # 每个epoch结束后的平均损失
            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
    def evaluate(self):
        """验证模型"""
        if not self.val_loader:
            return 0.0
            
        self.student.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                teacher_outputs = self.teacher(input_ids)
                student_outputs = self.student(input_ids)
                
                loss, _, _ = self.compute_loss(
                    student_outputs[0],
                    teacher_outputs.logits,
                    labels
                )
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        self.student.train()
        return avg_loss
    
    def save_model(self, save_name):
        """保存模型"""
        output_dir = os.path.join(self.config.output_dir, save_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.student.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # 保存配置
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
            
        self.logger.info(f"模型保存至 {output_dir}") 