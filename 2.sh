python data_read_png.py --input_dir '/home/h/Desktop/dataaaaa/4/input'\
 --label_dir '/home/h/Desktop/dataaaaa/4/label'\
 --output_dir 'test2' --mode 2
python train.py --data_dir 'test2'\
 --ckpt_dir 'test2/ckpt'\
 --log_dir 'test2/log'
python test_folder.py --test_dir '/home/h/Desktop/dataaaaa/3/input'\
 --ckpt_dir 'test2/ckpt'\
 --result_dir 'test2/result'