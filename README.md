# rknn-toolkit2-utils
适用于rknn-toolkit2与rknpu2的各种小工具


### 1. rknntoolkit2-versionswitch.py

快速切换rknn-toolkit2版本的小工具。是不是碰到有些版本的rknn-toolkit2恰好不支持你的模型，想多试几个版本，但是又不想手动卸载安装？这个小工具可以帮助你快速切换rknn-toolkit2的版本。  

```log
> rknntoolkit2-versionswitch.py
可用版本:
1. 1.4.0+22dcfef4
2. 1.5.0+1fa95b5c
3. 1.5.2+b642f30c
4. 1.6.0+81f21f4d
5. 1.6.2b2+d0f49553
6. 1.6.2b3+eeda9fd5
7. 2.0.0b0+9bab5682
8. 2.0.0b7+9eb76099
9. 2.0.0b12+1fc94019
10. 2.0.0b14+6b04b4c1

请输入指定的编号：9

...

成功安装 rknn_toolkit2-2.0.0b12+1fc94019-cp38-cp38-linux_x86_64.whl！
```
使用方法：

1. 将rknn-toolkit2的whl安装包放在此Python脚本同目录下
2. 运行脚本，选择需要安装的版本
3. 快速切换版本
4. 可以链接到`/usr/local/bin`下，方便全局使用

- 脚本中写死了安装包为Python 3.8版本，如有需要请自行修改

### 2. rknntoolkit2-regtask-decode.py

解析转换模型时出现的`REGTASK: The bit width of field value exceeds the limit`这类错误信息，快速查看到底是哪里超出了范围。  
如: `E RKNN: [10:48:24.306] REGTASK: The bit width of field value exceeds the limit, target: v2, offset: 0x500c, shift = 0, limit: 0x1fff, value: 0x3fff // DPU_RDMA rdma_data_cube_width.width: value=16383 > 8191`

推荐用法(可以定位是哪个算子导致的问题):
```bash
export RKNN_LOG_LEVEL=999
rm convert.log
python ./convert.py >> convert.log 2>>convert.log
cat ./convert.log | python rknntoolkit2-regtask-decode.py > convert.decoded.log
```

