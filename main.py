if __name__ == "__main__":
    
    # ARGUMENTS PARSER
    p = argparse.ArgumentParser()
    
    
    p.add_argument("--run_name", type=str, default="DDPM_conditional", help='(글자)임의의 러닝 폴더 이름을 넣어주세요'
    p.add_argument("--epochs", type=int, default=300, help='(정수) 훈련 횟수를 정해주세요'
    p.add_argument("--batch_size", type=int, default=8, help='(정수)배치 사이즈를 정해주세요'
    p.add_argument("--image_size", type=int, default=64, help='(정수)이미지 사이지를 정해주세요'
    p.add_argument("--num_classes", type=int, default=10, help='(정수)클래스의 갯수를 넣어주세요'
    p.add_argument("--dataset_path", type=str, default=r"/content/Diffusion-Models-pytorch/datasets/Landscape_classifier_02/training", help='(글자)데이터셋 경로를 넣어주세요'
    p.add_argument("--device", type=str, default="cuda", help='(글자)GPU 로 돌릴지 CPU로 돌릴지 정해주세요!'
    p.add_argument("--lr", type=float, default=3e-4, help='(부동소수점)러닝레이트를 넣어주세요'
                    
    args = p.parse_args()
                   
    train_model(args)
