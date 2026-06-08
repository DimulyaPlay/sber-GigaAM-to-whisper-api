# GigaAM OpenAI-compatible STT server

Локальный сервер реализует совместимый с OpenAI Whisper endpoint:

```text
POST /v1/audio/transcriptions
```

Он принимает `multipart/form-data` с полями `file`, `model`, `language`,
`response_format` и возвращает ответ в форматах `json`, `text`,
`verbose_json`, `srt` или `vtt`.

## Запуск

```powershell
.\.venv\Scripts\python.exe main.py --server --host 127.0.0.1 --port 8000 --model v3_e2e_rnnt
```

Для GPU:

```powershell
.\.venv\Scripts\python.exe main.py --server --gpu --host 127.0.0.1 --port 8000
```

Если нужен локальный ключ:

```powershell
.\.venv\Scripts\python.exe main.py --server --server-api-key local-secret
```

Тогда клиент должен отправлять `Authorization: Bearer local-secret`.
Без `--server-api-key` сервер принимает любой API key или пустое поле.

## Настройка клиента External Whisper

- Endpoint: `http://127.0.0.1:8000/v1/audio/transcriptions`
- API Key: пусто, если сервер запущен без `--server-api-key`
- Model: `gigaam`, `whisper-1`, `v3_e2e_rnnt` или `v3_rnnt`
- Language: `Russian`

`whisper-1` оставлен как alias на `v3_e2e_rnnt`, чтобы клиенты, которые
жестко ожидают Whisper-модель, работали без отдельной настройки.

## Проверка

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/v1/models
curl.exe -X POST http://127.0.0.1:8000/v1/audio/transcriptions `
  -F "file=@C:\path\to\audio.wav" `
  -F "model=gigaam" `
  -F "language=ru" `
  -F "response_format=json"
```
