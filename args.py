import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='GMNER LLM Refinement and Evaluation')
    
    # 数据路径参数
    parser.add_argument('--img_path', type=str, default="../GMNER/Twitter10000_v2.0/whole_image/",
                        help='图像文件路径')
    parser.add_argument('--output_dir', type=str, default="./output",
                        help='LLM输出目录路径')
    parser.add_argument('--save_process_file', type=str, default="./output/Answer.txt",
                        help='LLM处理过程保存文件路径')
    parser.add_argument('--local_pre_path', type=str, default="./data/predictions_test_epoch_0.json",
                        help='local model uncertainty过滤后预测结果路径')
    parser.add_argument('--local_refine_path', type=str, default="./data/predictions_test_epoch_0.json",
                        help='local model待refine路径')
    parser.add_argument('--gt_pre_path', type=str, default="./data/predictions_test_epoch_0.json",
                        help='ground truth路径')
    parser.add_argument('--llm_refine_path', type=str, default="./output/refine_pre.json",
                        help='LLM精炼结果路径')
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['refine', 'evaluate'], default='evaluate',
                        help='运行模式: refine(仅精炼), evaluate(仅评估))')
    parser.add_argument('--merge', type=bool, default=False,
                        help='是否合并local model预测结果')

    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args 