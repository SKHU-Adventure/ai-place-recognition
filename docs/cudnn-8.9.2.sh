cd ~/Downloads
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz

tar -xvf cudnn-linux-x86_64-8.9.2.26_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-8.9.2.26_cuda12-archive
sudo cp include/cudnn* /usr/local/cuda-12.1/include
sudo cp lib/libcudnn* /usr/local/cuda-12.1/lib64
sudo chmod a+r /usr/local/cuda-12.1/include/cudnn.h /usr/local/cuda-12.1/lib64/libcudnn*
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn.so.8
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_train.so
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_train.so
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_train.so
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_adv_infer.so
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_cnn_infer.so
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn_ops_infer.so
sudo ln -sf /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn.so.8.9.2 /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn.so
sudo ldconfig -N -v $(sed 's/:/ /' <<< %LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
