# API 接口文档

## 概述

本文档描述了贸易预测服务的所有API接口。服务分为两部分：
- **SpringBoot 后端服务**：提供训练任务管理、**远程文件替换**和数据上传功能
- **Flask 预测服务**：提供模型预测功能（训练完成后自动部署）

**基础URL**：
- SpringBoot服务：`http://localhost:8080/api`
- Flask服务：`http://localhost:5000`

---

## 一、SpringBoot 后端 API

### 1. 启动训练任务

启动一个完整的训练任务，系统会自动按顺序执行以下步骤：
1. 删除 `python-scripts/model` 文件夹（如果存在）
2. 训练进口数量模型
3. 训练进口单价模型
4. 训练出口数量模型
5. 训练出口单价模型
6. 自动部署 Flask API 服务
7. 任务完成

**接口地址**：`POST /api/training/start`

**请求方式**：`POST`

**请求参数**：无

**请求示例**：
```bash
curl -X POST http://localhost:8080/api/training/start
```

**响应示例**：
```json
{
  "success": true,
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "message": "训练任务已启动"
}
```

**错误响应**：
```json
{
  "success": false,
  "message": "启动训练任务失败: [错误信息]"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| success | boolean | 是否成功 |
| taskId | string | 任务ID，用于后续查询任务状态 |
| message | string | 响应消息 |

---

### 2. 查询指定任务状态

根据任务ID查询训练任务的详细状态信息。

**接口地址**：`GET /api/training/status/{taskId}`

**请求方式**：`GET`

**路径参数**：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| taskId | string | 是 | 任务ID，从启动训练任务接口返回 |

**请求示例**：
```bash
curl http://localhost:8080/api/training/status/550e8400-e29b-41d4-a716-446655440000
```

**响应示例**（运行中）：
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "overallStatus": "RUNNING",
  "startTime": "2024-01-01T10:00:00",
  "endTime": null,
  "totalDuration": null,
  "currentStep": 2,
  "totalSteps": 6,
  "progress": 33.33,
  "scriptResults": [
    {
      "scriptName": "Trade_Transformer_LSTM_quantity.py",
      "status": "SUCCESS",
      "startTime": "2024-01-01T10:00:00",
      "endTime": "2024-01-01T10:30:00",
      "duration": 1800,
      "errorMessage": null,
      "output": "训练完成..."
    }
  ],
  "errorMessage": null
}
```

**响应示例**（已完成）：
```json
{
  "taskId": "550e8400-e29b-41d4-a716-446655440000",
  "overallStatus": "SUCCESS",
  "startTime": "2024-01-01T10:00:00",
  "endTime": "2024-01-01T12:00:00",
  "totalDuration": 7200,
  "currentStep": 6,
  "totalSteps": 6,
  "progress": 100.0,
  "scriptResults": [
    {
      "scriptName": "Trade_Transformer_LSTM_quantity.py",
      "status": "SUCCESS",
      "startTime": "2024-01-01T10:00:00",
      "endTime": "2024-01-01T10:30:00",
      "duration": 1800,
      "errorMessage": null,
      "output": "..."
    },
    {
      "scriptName": "Trade_Transformer_LSTM_price.py",
      "status": "SUCCESS",
      "startTime": "2024-01-01T10:30:00",
      "endTime": "2024-01-01T11:00:00",
      "duration": 1800,
      "errorMessage": null,
      "output": "..."
    },
    {
      "scriptName": "Trade_Transformer_LSTM_quantity.py",
      "status": "SUCCESS",
      "startTime": "2024-01-01T11:00:00",
      "endTime": "2024-01-01T11:30:00",
      "duration": 1800,
      "errorMessage": null,
      "output": "..."
    },
    {
      "scriptName": "Trade_Transformer_LSTM_price.py",
      "status": "SUCCESS",
      "startTime": "2024-01-01T11:30:00",
      "endTime": "2024-01-01T12:00:00",
      "duration": 1800,
      "errorMessage": null,
      "output": "..."
    },
    {
      "scriptName": "app.py",
      "status": "SUCCESS",
      "startTime": "2024-01-01T12:00:00",
      "endTime": "2024-01-01T12:00:30",
      "duration": 30,
      "errorMessage": null,
      "output": "Flask服务已启动..."
    }
  ],
  "errorMessage": null
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| taskId | string | 任务ID |
| overallStatus | string | 总体状态：PENDING（等待）、RUNNING（运行中）、SUCCESS（成功）、FAILED（失败）、CANCELLED（已取消） |
| startTime | string | 任务开始时间（ISO 8601格式） |
| endTime | string | 任务结束时间（ISO 8601格式），未完成时为null |
| totalDuration | long | 总耗时（秒），未完成时为null |
| currentStep | integer | 当前执行的步骤（1-6） |
| totalSteps | integer | 总步骤数（固定为6） |
| progress | double | 进度百分比（0.0-100.0） |
| scriptResults | array | 各步骤的执行结果列表 |
| errorMessage | string | 错误信息，无错误时为null |

**scriptResults 字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| scriptName | string | 脚本名称 |
| status | string | 执行状态：PENDING、RUNNING、SUCCESS、FAILED、CANCELLED |
| startTime | string | 开始时间 |
| endTime | string | 结束时间 |
| duration | long | 执行耗时（秒） |
| errorMessage | string | 错误信息 |
| output | string | 脚本输出信息 |

**HTTP状态码**：
- `200 OK`：查询成功
- `404 Not Found`：任务不存在

---

### 3. 查询所有任务状态

获取所有训练任务的状态列表。

**接口地址**：`GET /api/training/status`

**请求方式**：`GET`

**请求参数**：无

**请求示例**：
```bash
curl http://localhost:8080/api/training/status
```

**响应示例**：
```json
[
  {
    "taskId": "550e8400-e29b-41d4-a716-446655440000",
    "overallStatus": "SUCCESS",
    "startTime": "2024-01-01T10:00:00",
    "endTime": "2024-01-01T12:00:00",
    "totalDuration": 7200,
    "currentStep": 6,
    "totalSteps": 6,
    "progress": 100.0,
    "scriptResults": [...],
    "errorMessage": null
  },
  {
    "taskId": "660e8400-e29b-41d4-a716-446655440001",
    "overallStatus": "RUNNING",
    "startTime": "2024-01-01T13:00:00",
    "endTime": null,
    "totalDuration": null,
    "currentStep": 2,
    "totalSteps": 6,
    "progress": 33.33,
    "scriptResults": [...],
    "errorMessage": null
  }
]
```

**响应字段说明**：同"查询指定任务状态"接口

---

### 4. 远程文件替换接口

**远程上传并替换训练数据文件**。通过此接口可以远程上传新的训练数据文件，替换服务器上现有的 `merged_input.csv` 和/或 `merged_output.csv` 文件，无需手动登录服务器操作。

---

### 5. 上传进口数据到FastGPT

**专用上传接口，用于上传进口CSV数据到FastGPT平台**。

**接口地址**：`POST /api/upload/to-fastgpt/import`

**请求方式**：`POST`

**Content-Type**：`multipart/form-data`

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| file | file | 是 | 要上传的进口CSV文件 |

**功能说明**：
- 专为进口数据设计的上传接口
- 自动设置数据类型为进口数据
- 简化上传流程，无需手动配置
- 固定使用进口数据集ID：68e8cb47a3d85a7f250333cb

**请求示例**（Windows PowerShell）：
```powershell
curl -X POST http://localhost:8080/api/upload/to-fastgpt/import `
  -F "file=@D:/tmp/import_data.csv"
```

**请求示例**（Linux/Mac）：
```bash
curl -X POST http://localhost:8080/api/upload/to-fastgpt/import \
  -F "file=@/tmp/import_data.csv"
```

**响应示例**：
```json
{
  "success": true,
  "message": "进口数据上传FastGPT成功",
  "data": {
    "id": "68e8cb47a3d85a7f250333cb",
    "name": "进口数据集"
  }
}
```

**错误响应**：
```json
{
  "error": "文件上传失败",
  "message": "[具体错误信息]"
}
```

**HTTP状态码**：
- `200 OK`：上传成功
- `400 Bad Request`：参数错误或文件格式不正确
- `500 Internal Server Error`：服务器内部错误

---

### 6. 上传出口数据到FastGPT

**专用上传接口，用于上传出口CSV数据到FastGPT平台**。

**接口地址**：`POST /api/upload/to-fastgpt/export`

**请求方式**：`POST`

**Content-Type**：`multipart/form-data`

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| file | file | 是 | 要上传的出口CSV文件 |

**功能说明**：
- 专为出口数据设计的上传接口
- 自动设置数据类型为出口数据
- 简化上传流程，无需手动配置
- 固定使用出口数据集ID：68e8ce82a3d85a7f25042962

**请求示例**（Windows PowerShell）：
```powershell
curl -X POST http://localhost:8080/api/upload/to-fastgpt/export `
  -F "file=@D:/tmp/export_data.csv"
```

**请求示例**（Linux/Mac）：
```bash
curl -X POST http://localhost:8080/api/upload/to-fastgpt/export \
  -F "file=@/tmp/export_data.csv"
```

**响应示例**：
```json
{
  "success": true,
  "message": "出口数据上传FastGPT成功",
  "data": {
    "id": "68e8ce82a3d85a7f25042962",
    "name": "出口数据集"
  }
}
```

**错误响应**：
```json
{
  "error": "文件上传失败",
  "message": "[具体错误信息]"
}
```

**HTTP状态码**：
- `200 OK`：上传成功
- `400 Bad Request`：参数错误或文件格式不正确
- `500 Internal Server Error`：服务器内部错误

**接口地址**：`POST /api/data/upload`

**请求方式**：`POST`

**Content-Type**：`multipart/form-data`

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| mergedInput | file | 否 | 新的 `merged_input.csv` 文件（进口数据） |
| mergedOutput | file | 否 | 新的 `merged_output.csv` 文件（出口数据） |

**功能说明**：
- 此接口支持**远程文件替换**，无需登录服务器即可更新训练数据
- 至少需要上传一个文件（可以只上传 `mergedInput` 或只上传 `mergedOutput`）
- 文件必须是有效的CSV格式
- 文件会被自动保存到配置文件中指定的路径，替换原有文件
- 替换后的文件会在下次训练任务时自动使用

**请求示例**（Windows PowerShell）：
```powershell
curl -X POST http://localhost:8080/api/data/upload `
  -F "mergedInput=@D:/tmp/merged_input.csv" `
  -F "mergedOutput=@D:/tmp/merged_output.csv"
```

**请求示例**（Linux/Mac）：
```bash
curl -X POST http://localhost:8080/api/data/upload \
  -F "mergedInput=@/tmp/merged_input.csv" \
  -F "mergedOutput=@/tmp/merged_output.csv"
```

**响应示例**：
```json
{
  "success": true,
  "message": "数据文件上传成功",
  "files": [
    {
      "alias": "merged_input",
      "originalFilename": "merged_input.csv",
      "size": 123456,
      "savedPath": "D:/TradeSpringBoot/data/进口/merged_input.csv"
    },
    {
      "alias": "merged_output",
      "originalFilename": "merged_output.csv",
      "size": 234567,
      "savedPath": "D:/TradeSpringBoot/data/出口/merged_output.csv"
    }
  ]
}
```

**错误响应**：
```json
{
  "success": false,
  "message": "至少需要上传一个文件"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| success | boolean | 是否成功 |
| message | string | 响应消息 |
| files | array | 上传成功的文件列表 |

**files 字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| alias | string | 文件别名（merged_input 或 merged_output） |
| originalFilename | string | 原始文件名 |
| size | long | 文件大小（字节） |
| savedPath | string | 保存路径 |

**HTTP状态码**：
- `200 OK`：上传成功
- `400 Bad Request`：请求参数错误

---

## 二、Flask 预测服务 API

Flask服务在训练任务完成后自动部署，默认运行在 `http://localhost:5000`。

### 1. 健康检查

检查Flask服务是否正常运行。

**接口地址**：`GET /health`

**请求方式**：`GET`

**请求参数**：无

**请求示例**：
```bash
curl http://localhost:5000/health
```

**响应示例**：
```json
{
  "status": "ok",
  "timestamp": "2024-01-01T10:00:00.000000"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| status | string | 服务状态，固定为 "ok" |
| timestamp | string | 当前时间戳（ISO 8601格式） |

---

### 2. 预测进口单价

根据历史数据预测进口商品的单价。

**接口地址**：`POST /predict_in_price`

**请求方式**：`POST`

**Content-Type**：`application/json`

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| 贸易伙伴名称 | string | 是 | 贸易伙伴名称 |
| 商品名称 | string | 是 | 商品名称 |
| 贸易方式 | string | 是 | 贸易方式名称 |
| 注册地名称 | string | 是 | 注册地名称 |
| year | integer | 是 | 目标年份 |
| month | integer | 是 | 目标月份（1-12） |

**请求示例**：
```bash
curl -X POST http://localhost:5000/predict_in_price \
  -H "Content-Type: application/json" \
  -d '{
    "贸易伙伴名称": "美国",
    "商品名称": "电子产品",
    "贸易方式": "一般贸易",
    "注册地名称": "北京",
    "year": 2024,
    "month": 3
  }'
```

**响应示例**：
```json
{
  "predicted_in_price": 1234.56,
  "unit": "人民币/千克"
}
```

**响应示例**（历史数据不足时）：
```json
{
  "predicted_in_price": 1234.56,
  "unit": "人民币/千克",
  "warning": "历史数据不足12条，预测结果准确度受影响"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| predicted_in_price | float | 预测的进口单价 |
| unit | string | 单位（格式：人民币/计量单位） |
| warning | string | 警告信息（可选），当历史数据不足时返回 |

**HTTP状态码**：
- `200 OK`：预测成功
- `400 Bad Request`：请求参数错误或数据不足

**错误响应示例**：
```json
{
  "error": "无效的年/月"
}
```

---

### 3. 预测进口数量

根据历史数据预测进口商品的数量。

**接口地址**：`POST /predict_in_quantity`

**请求方式**：`POST`

**Content-Type**：`application/json`

**请求参数**：同"预测进口单价"接口

**请求示例**：
```bash
curl -X POST http://localhost:5000/predict_in_quantity \
  -H "Content-Type: application/json" \
  -d '{
    "贸易伙伴名称": "美国",
    "商品名称": "电子产品",
    "贸易方式": "一般贸易",
    "注册地名称": "北京",
    "year": 2024,
    "month": 3
  }'
```

**响应示例**：
```json
{
  "predicted_in_quantity": 5678.90,
  "unit": "千克"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| predicted_in_quantity | float | 预测的进口数量 |
| unit | string | 计量单位 |
| warning | string | 警告信息（可选） |

---

### 4. 预测出口单价

根据历史数据预测出口商品的单价。

**接口地址**：`POST /predict_out_price`

**请求方式**：`POST`

**Content-Type**：`application/json`

**请求参数**：同"预测进口单价"接口

**请求示例**：
```bash
curl -X POST http://localhost:5000/predict_out_price \
  -H "Content-Type: application/json" \
  -d '{
    "贸易伙伴名称": "美国",
    "商品名称": "电子产品",
    "贸易方式": "一般贸易",
    "注册地名称": "北京",
    "year": 2024,
    "month": 3
  }'
```

**响应示例**：
```json
{
  "predicted_out_price": 2345.67,
  "unit": "人民币/千克"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| predicted_out_price | float | 预测的出口单价 |
| unit | string | 单位（格式：人民币/计量单位） |
| warning | string | 警告信息（可选） |

---

### 5. 预测出口数量

根据历史数据预测出口商品的数量。

**接口地址**：`POST /predict_out_quantity`

**请求方式**：`POST`

**Content-Type**：`application/json`

**请求参数**：同"预测进口单价"接口

**请求示例**：
```bash
curl -X POST http://localhost:5000/predict_out_quantity \
  -H "Content-Type: application/json" \
  -d '{
    "贸易伙伴名称": "美国",
    "商品名称": "电子产品",
    "贸易方式": "一般贸易",
    "注册地名称": "北京",
    "year": 2024,
    "month": 3
  }'
```

**响应示例**：
```json
{
  "predicted_out_quantity": 8901.23,
  "unit": "千克"
}
```

**响应字段说明**：

| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| predicted_out_quantity | float | 预测的出口数量 |
| unit | string | 计量单位 |
| warning | string | 警告信息（可选） |

---

## 三、状态码说明

### 任务状态（overallStatus / status）

| 状态值 | 说明 |
| --- | --- |
| PENDING | 等待执行 |
| RUNNING | 运行中 |
| SUCCESS | 执行成功 |
| FAILED | 执行失败 |
| CANCELLED | 已取消 |

### HTTP状态码

| 状态码 | 说明 |
| --- | --- |
| 200 | 请求成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

---

## 四、使用流程示例

### 完整训练和预测流程

1. **远程替换训练数据文件**（可选，如需更新训练数据）
   - 使用远程文件替换接口，无需登录服务器即可更新数据
```bash
curl -X POST http://localhost:8080/api/data/upload \
  -F "mergedInput=@merged_input.csv" \
  -F "mergedOutput=@merged_output.csv"
```

2. **启动训练任务**
```bash
curl -X POST http://localhost:8080/api/training/start
# 返回: {"success": true, "taskId": "xxx", "message": "训练任务已启动"}
```

3. **查询训练进度**（轮询）
```bash
curl http://localhost:8080/api/training/status/xxx
# 持续查询直到 overallStatus 为 SUCCESS
```

4. **训练完成后，使用Flask API进行预测**
```bash
# 健康检查
curl http://localhost:5000/health

# 预测进口单价
curl -X POST http://localhost:5000/predict_in_price \
  -H "Content-Type: application/json" \
  -d '{
    "贸易伙伴名称": "美国",
    "商品名称": "电子产品",
    "贸易方式": "一般贸易",
    "注册地名称": "北京",
    "year": 2024,
    "month": 3
  }'
```

---

## 五、注意事项

1. **训练任务**：
   - 每次启动训练任务时，系统会自动删除 `python-scripts/model` 文件夹
   - 训练任务异步执行，不会阻塞请求
   - 训练可能需要较长时间，建议定期查询任务状态

2. **Flask服务**：
   - Flask服务在训练完成后自动启动
   - 默认运行在 `http://localhost:5000`
   - 如果端口被占用，可以在配置文件中修改端口

3. **数据要求**：
   - 预测接口需要从数据库查询历史数据
   - 历史数据不足12条时，会返回警告信息
   - 确保数据库连接配置正确

4. **错误处理**：
   - 所有接口都会返回明确的错误信息
   - 建议在生产环境中添加适当的错误处理和重试机制

---

## 六、常见问题

**Q: 训练任务一直处于RUNNING状态？**
A: 检查日志文件 `logs/trade-service.log`，查看是否有错误信息。训练可能需要较长时间，请耐心等待。

**Q: Flask服务无法访问？**
A: 确保训练任务已完成，并且Flask服务已成功启动。检查端口5000是否被占用。

**Q: 预测接口返回错误？**
A: 检查请求参数是否正确，确保数据库中有对应的历史数据。

**Q: 如何修改Flask服务端口？**
A: 修改 `application.yml` 中的 `python.flask.port` 配置项。

---

## 更新日志

- **2024-01-01**：初始版本
  - 添加训练任务管理接口
  - 添加数据上传接口
  - 添加Flask预测服务接口
  - 添加自动删除model文件夹功能

