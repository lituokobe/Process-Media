## API 接口
### 健康检查 
GET http://ai-director.f1.luyouxia.net:17523/health (本地：http://localhost:8013/health)  
输出:
```json
{
    "status": "healthy",
    "service": "process-media",
    "timestamp": "2026-05-18T09:52:47.726329"
}
```

### 处理数据 
POST http://ai-director.f1.luyouxia.net:17523/process (本地：http://localhost:8013/process)  
输入：
```json
{
    "data_path": "数据文件路径"
}
```
输出：
```json
{
	"type": "entry" | "connected" | "progress" | "complete", # 只有"entry"时，下方的"data"是录入向量数据库的数据
	"data": {
		"org_id": 46,
		"material_id": 5,
		"material_path": "https://freepd.cn/api/music/53636f72696e672f416374696f6e20537472696b652e6d7033.mp3",
		"desc_json": {
			"overall_summary": "这段音乐是轻快的流行风格，以钢琴和弦乐为主导乐器，配以柔和的打击乐。音乐的旋律简洁明快，给人一种轻松愉悦的感觉，适合在商业、会展等场合使用，以营造轻松愉快的氛围。",
			"duration": 134.557
		},
		"version": "1.3"
	},
	"timestamp": "2026-05-22T14:10:57.966759"
}
```
### 测试
```shell
curl -N -X GET http://ai-director.f1.luyouxia.net:17523/health
```
```shell
curl -N -X POST http://ai-director.f1.luyouxia.net:17523/stream_process -H "Content-Type: application/json" -d '{"data_path": ""}'
```