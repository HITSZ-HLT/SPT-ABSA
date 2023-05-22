## 环境配置

- transformers==4.15.0
- pytorch==1.7.1
- pytorch-lightning==1.3.5
- datasets==2.8.0

## 运行代码

配置好路径，运行一下命令：
- `bash/400k.sh -c 0,1 -d {your_data_dir} -o {your_output_dir} -c {your_cache_dir}` 

注：当前的batch_size是在A100(40G)设置的，请根据显存大小自行调整

`
