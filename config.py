from dataclasses import dataclass

@dataclass
class DistillationConfig:
    # 教师模型配置
    teacher_model_type: str = 'gpt2-medium'  # 使用较大的GPT2模型作为教师
    
    # 学生模型配置
    student_n_layer: int = 6  # 教师模型层数的一半
    student_n_head: int = 8   # 较少的注意力头
    student_n_embd: int = 384 # 较小的嵌入维度
    
    # 训练配置
    batch_size: int = 32
    seq_length: int = 128
    max_steps: int = 20000
    warmup_steps: int = 1000
    
    # 优化器配置
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    
    # 蒸馏配置
    temperature: float = 2.0   # 软标签的温度系数
    alpha: float = 0.5        # 蒸馏损失和任务损失的权重
    
    # 其他配置
    seed: int = 42
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    output_dir: str = "distilled_model" 
    
    # 添加新的配置
    num_epochs: int = 3
    max_length: int = 512  # 可以根据需要调整序列长度 