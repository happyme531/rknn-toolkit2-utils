#!/usr/bin/env python3

import os
import re
import sys
import subprocess
from natsort import natsorted

# 默认文件夹路径
DEFAULT_DIRECTORY = os.path.realpath(__file__).rsplit('/', 1)[0]

# 获取安装包版本的正则表达式
PACKAGE_REGEX = r'rknn_toolkit2-([0-9a-zA-Z\.\+\-]+)-cp38-cp38-linux_x86_64\.whl'

def find_packages(directory):
    """在指定目录中查找符合正则表达式的安装包"""
    packages = {}
    for filename in os.listdir(directory):
        match = re.match(PACKAGE_REGEX, filename)
        if match:
            version = match.group(1)
            packages[version] = filename
    return packages

def select_version(packages):
    """显示可用版本并让用户选择"""
    sorted_versions = natsorted(packages.keys())
    print("可用版本:")
    for index, version in enumerate(sorted_versions, start=1):
        print(f"{index}. {version}")
    
    while True:
        try:
            choice = int(input("\n请输入指定的编号："))
            if 1 <= choice <= len(sorted_versions):
                return sorted_versions[choice - 1]
            else:
                print("无效的编号，请输入有效的编号。")
        except ValueError:
            print("无效的输入，请输入有效的编号。")

def install_package(directory, package_filename):
    """使用 pip 安装指定的安装包"""
    package_path = os.path.join(directory, package_filename)
    try:
        subprocess.run(['pip', 'install', '--no-deps', package_path], check=True)
        print(f"\n成功安装 {package_filename}！")
    except subprocess.CalledProcessError as e:
        print(f"\n错误：安装 {package_filename} 失败！\n{e}")

def main(directory):
    """主函数"""
    if not os.path.isdir(directory):
        print(f"错误：指定的目录不存在：{directory}")
        sys.exit(1)

    packages = find_packages(directory)

    if not packages:
        print(f"错误：在目录 {directory} 中未找到符合条件的安装包。")
        sys.exit(1)

    selected_version = select_version(packages)
    selected_package = packages[selected_version]

    install_package(directory, selected_package)

if __name__ == "__main__":
    # 获取用户指定的目录或使用默认目录
    user_directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIRECTORY
    main(user_directory)
