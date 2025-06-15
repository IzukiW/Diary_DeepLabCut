TOC

## 25/06/12 

task
- 必要な教師データのフレーム数の把握
- 画像処理の下限

**Done**
1. anaconda上に仮想環境DEEPLABCUTを作成
1. deeplabcutに必要なcuda12.6(local), cuDNN(仮想環境)等をinstall
1. CUDAのpath通し

GPU version確認 >`nvidia-smi`
CUDA version確認 >`nvcc -V`



**Trouble**
1. `python -m deeplabcut` で起動しない -> 仮想環境(DEEPLABCUT)にpipがinstallされていなかった -> installしたら謎エラーを吐かれた
<details><summary>error表示1 FileNotFoundError"などとほざいている </summary>
```
>pip uninstall numpy
Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\Scripts\pip-script.py", line 9, in <module>
    sys.exit(main())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main.py", line 64, in main
    cmd_name, cmd_args = parse_command(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main_parser.py", line 78, in parse_command
    general_options, args_else = parser.parse_args(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\optparse.py", line 1371, in parse_args
    values = self.get_default_values()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\parser.py", line 279, in get_default_values
    self.config.load()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 124, in load
    self._load_config_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 246, in _load_config_files
    config_files = dict(self.iter_config_files())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 339, in iter_config_files
    config_files = get_configuration_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 70, in get_configuration_files
    os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs("pip")
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\utils\appdirs.py", line 48, in site_config_dirs
    dirval = _appdirs.site_config_dir(appname, appauthor=False, multipath=True)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\__init__.py", line 146, in site_config_dir
    ).site_config_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 67, in site_config_dir
    return self.site_data_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 56, in site_data_dir
    path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 209, in get_win_folder_from_registry
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
FileNotFoundError: [WinError 2] The system cannot find the file specified
```
</details>


`conda install pip`したのに治らない
<details><summary>error表示2 同じ表示だね </summary>
```
>pip --verison
Traceback (most recent call last):
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\Scripts\pip-script.py", line 9, in <module>
    sys.exit(main())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main.py", line 64, in main
    cmd_name, cmd_args = parse_command(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\main_parser.py", line 78, in parse_command
    general_options, args_else = parser.parse_args(args)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\optparse.py", line 1371, in parse_args
    values = self.get_default_values()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\cli\parser.py", line 279, in get_default_values
    self.config.load()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 124, in load
    self._load_config_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 246, in _load_config_files
    config_files = dict(self.iter_config_files())
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 339, in iter_config_files
    config_files = get_configuration_files()
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\configuration.py", line 70, in get_configuration_files
    os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs("pip")
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_internal\utils\appdirs.py", line 48, in site_config_dirs
    dirval = _appdirs.site_config_dir(appname, appauthor=False, multipath=True)
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\__init__.py", line 146, in site_config_dir
    ).site_config_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 67, in site_config_dir
    return self.site_data_dir
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 56, in site_data_dir
    path = os.path.normpath(get_win_folder("CSIDL_COMMON_APPDATA"))
  File "C:\Users\satie\anaconda3\envs\DEEPLABCUT\lib\site-packages\pip\_vendor\platformdirs\windows.py", line 209, in get_win_folder_from_registry
    directory, _ = winreg.QueryValueEx(key, shell_folder_name)
FileNotFoundError: [WinError 2] The system cannot find the file specified
```
</details>

(DEEPLABCUT)上のpipをuninstallしたらエラーが消えた。baseのpipとconflictしてたのだろうか？


## 2025/06/13

昨日は通らなかった`nvcc -V`が通るようになった。
```
C:\Users\satie>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:44:19_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
```

**Done**
- pytorchのinstall
- deeplabcut GUI版+model zooのinstall
- GUIでlabeling(manubrium_baseだけできなかったけど...)

installしたPytorch
![PyTorch version](./fig/Screenshot_13-6-2025_1990_pytorch.org.jpeg)

install後公式文書通りの応答が帰ってきた
```
#in my env(DEEPLABCUT)

(DEEPLABCUT) C:\Users\satie>python -c "import torch; print(torch.cuda.is_available())"
True
```

DeepLabCutは`pip install deeplabcut[gui,modelzoo]`で仮想環境上に置いた



[公式ドキュメント](https://deeplabcut.github.io/DeepLabCut/docs/standardDeepLabCut_UserGuide.html)にならって
```py
ipython
import deeplabcut
```
すると"ModuleNotFoundError"が次々出てくる
<details><summary>error表示3 ruamel.yamlがありません </summary>

```python
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\deeplabcut\utils\auxiliaryfunctions.py:31
     29 import numpy as np
     30 import pandas as pd
---> 31 import ruamel.yaml.representer
     32 import yaml
     33 from ruamel.yaml import YAML

ModuleNotFoundError: No module named 'ruamel'
```
</details>

`conda install ruamel.yaml` で対処

<details><summary>error表示 filelockがありません </summary>

```python
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\huggingface_hub\utils\_fixes.py:22
     19 from typing import Callable, Generator, Optional, Union
     21 import yaml
---> 22 from filelock import BaseFileLock, FileLock, SoftFileLock, Timeout
     24 from .. import constants
     25 from . import logging

ModuleNotFoundError: No module named 'filelock'
```
</details>

`conda install conda-forge::filelock` で対処

<details><summary>error表示5 sympyがinstallできません </summary>

```python
File ~\anaconda3\envs\DEEPLABCUT\lib\site-packages\torch\utils\_sympy\functions.py:10
      7 from typing_extensions import TypeVarTuple, Unpack
      9 import sympy
---> 10 from sympy import S
     11 from sympy.core import sympify
     12 from sympy.core.expr import Expr

ImportError: cannot import name 'S' from 'sympy' (unknown location)
```
</details>

`conda install sympy` で対処

エラーが出なくなった！！
anacondaからDeepLabCut(GUI)の起動方法は2つ
```
ipython
import deeplabcut
deeplabcut.launch_dlc()
```
または
```
python -m deeplabcut
```
![DeepLabCutの起動画面](./fig/Screenshot_2025-06-13_201301.png)

GUIにしたがってprojectを作成

![alt text](./fig/Screenshot_2025-06-13_202642.png)


## 25/06/15

**Done**
- DSC_3189_#1~5, 3197_#1~2のlabeling 
(4points: manubrium base, mouth, prey, tentacle base)
- DSC_3197/3237_#1~5のframe抽出
- dlc3.0の[Tutorial](https://youtu.be/ofFx0vTMSxE?si=hsLphcDQQz36v3cA)を見る(途中)
- `Create training dataset`タブでラベル済みの140データからなるdatasetを作成
- `Train Network`を実行


前回の続きなので、トップ画面から`Load Project`を選択 > config.yamlをopen

1video当たり20frames抽出する設定
20framesに4point打つのにかかった時間: 10分弱

Trainの様子


![Training中](./Fig/Screenshot_2025-06-15_222717.png)


