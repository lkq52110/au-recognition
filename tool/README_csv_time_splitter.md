# CSV Time Splitter Tool

This tool processes CSV files and splits them into 8 time periods based on the 'time' column (timestamp in milliseconds).

## 功能特性 (Features)

- 自动扫描当前目录下的所有CSV文件
- 根据时间戳将数据分为8个预定义的时间段
- 创建对应的测试文件夹("测试1"到"测试8")
- 保存过滤后的数据，文件名添加"测试x"后缀
- 提供详细的处理日志和错误处理
- 支持多种文件编码格式(UTF-8, GBK, GB2312, Latin-1)

## 时间段定义 (Time Period Definitions)

| 测试编号 | 时间范围 | 毫秒范围 |
|---------|---------|---------|
| 测试1 | 00:00~02:38 | 0-158,000ms |
| 测试2 | 02:39~04:33 | 159,000-273,000ms |
| 测试3 | 04:34~05:18 | 274,000-318,000ms |
| 测试4 | 05:19~06:46 | 319,000-406,000ms |
| 测试5 | 06:47~08:03 | 407,000-483,000ms |
| 测试6 | 08:06~09:13 | 486,000-553,000ms |
| 测试7 | 09:14~09:52 | 554,000-592,000ms |
| 测试8 | 09:52~10:46 | 592,000-646,000ms |

## 使用方法 (Usage)

### 前提条件 (Prerequisites)

确保已安装pandas库:
```bash
pip install pandas
```

### 运行脚本 (Running the Script)

1. 将要处理的CSV文件放在当前目录下
2. 确保CSV文件包含'time'列（时间戳，毫秒单位）
3. 运行脚本:

```bash
python tool/csv_time_splitter.py
```

### 输入要求 (Input Requirements)

- CSV文件必须包含名为'time'的列
- 'time'列的值应为毫秒级时间戳
- 文件应该是有效的CSV格式

### 输出结果 (Output)

脚本将会:
1. 创建8个文件夹：测试1, 测试2, ..., 测试8
2. 在每个文件夹中保存对应时间段的数据
3. 生成的文件名格式：`原文件名_测试x.csv`
4. 在控制台和日志文件中显示详细的处理信息

## 示例 (Example)

假设有一个名为`data.csv`的文件：
```csv
time,value,label
50000,1.5,A
200000,2.3,B
350000,3.1,C
```

运行脚本后会产生：
- `测试1/data_测试1.csv` (包含time=50000的数据)
- `测试2/data_测试2.csv` (包含time=200000的数据)
- `测试4/data_测试4.csv` (包含time=350000的数据)

## 错误处理 (Error Handling)

脚本会处理以下情况：
- 缺少'time'列的CSV文件
- 文件编码问题
- 空文件或损坏的CSV文件
- 文件读取权限问题

所有错误信息和处理状态都会记录在控制台和`csv_processing.log`日志文件中。

## 日志文件 (Log File)

脚本会生成`csv_processing.log`文件，记录：
- 处理进度信息
- 错误和警告消息
- 每个时间段的数据统计
- 最终处理结果汇总