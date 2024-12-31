import torch
from config import DistillationConfig
from student_model import StudentTransformer, StudentConfig
from distiller import DistillationTrainer
from torch.utils.data import Dataset
import tiktoken
from datasets import load_dataset

class AlpacaDataset(Dataset):
    def __init__(self, max_length=128):
        # 加载数据集
        self.dataset = load_dataset(
            "json",
            data_files="../dataset/ruozhiba_qa.json",  # 使用相对路径
            split="train"
        )
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.max_length = max_length
        
        # Alpaca 格式模板
        self.prompt_template = """Below is an instruction that describes a task, paired with an input that provides further collaboration details.

### Instruction:
{}
### Input:
{}
### Response:
{}"""

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # 格式化文本
        text = self.prompt_template.format(
            example['instruction'],
            example['input'],
            example['output']
        )
        
        # 编码文本
        tokens = self.tokenizer.encode(text)
        
        # 如果序列太长，进行截断
        if len(tokens) > self.max_length + 1:
            tokens = tokens[:self.max_length + 1]
        
        # 如果序列太短，进行填充
        elif len(tokens) < self.max_length + 1:
            tokens = tokens + [self.tokenizer.eot_token] * (self.max_length + 1 - len(tokens))
            
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }

def main():
    # 设置配置
    config = DistillationConfig()
    
    # 创建学生模型
    student_config = StudentConfig(
        n_layer=config.student_n_layer,
        n_head=config.student_n_head,
        n_embd=config.student_n_embd
    )
    student_model = StudentTransformer(student_config)
    
    # 创建训练集和验证集
    full_dataset = AlpacaDataset(max_length=config.seq_length)
    
    # 划分训练集和验证集 (90% 训练, 10% 验证)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size]
    )
    
    # 创建训练器
    trainer = DistillationTrainer(
        config=config,
        student_model=student_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main() 