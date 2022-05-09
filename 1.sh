python data_read_png.py --input_dir '/home/h/Desktop/dataaaaa/4/input'\
 --label_dir '/home/h/Desktop/dataaaaa/4/label'\
 --output_dir 'test1' 
python train.py --data_dir 'test1'\
 --ckpt_dir 'test1/ckpt'\
 --log_dir 'test1/log'
python eval.py --data_dir 'test1' --ckpt_dir 'test1/ckpt' --result_dir 'test1/result' 
