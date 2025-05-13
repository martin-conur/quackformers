# Quackformers FastAPI Server

This server provides an API for embedding text using the Quackformers DuckDB extension. It supports two endpoints: `/embed` and `/embed_jina`.

## Prerequisites

1. **Install Python dependencies**:
   Ensure you have Python 3.7+ installed. Install the required dependencies using `pip`:
   ```bash
   pip install fastapi uvicorn duckdb pydantic
   ```

2. **Install the Quackformers extension**:
   Make sure the Quackformers DuckDB extension is installed and accessible.

## Running the Server

1. Start the server by running the `server.py` script:
   ```bash
   python server.py
   ```

2. The server will start on `http://0.0.0.0:8080/`.

## Endpoints

### 1. `/embed`

- **Description**: Embeds the input text using the `embed` function from the Quackformers extension.
- **Method**: `POST`
- **Request Payload**:
  ```json
  {
    "text": "This is a sample sentence."
  }
  ```
- **Response**:
  ```json
  {
    "embedded_text": [0.123, 0.456, 0.789, ...]  // Example embedding values
  }
  ```

- **Example `curl` Command**:
  ```bash
  curl -X POST "http://0.0.0.0:8080/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample sentence."}'
  ```

### 2. `/embed_jina`

- **Description**: Embeds the input text using the `embed_jina` function from the Quackformers extension.
- **Method**: `POST`
- **Request Payload**:
  ```json
  {
    "text": "This is another sample sentence."
  }
  ```
- **Response**:
  ```json
  {
    "embedded_text": [0.987, 0.654, 0.321, ...]  // Example embedding values
  }
  ```

- **Example `curl` Command**:
  ```bash
  curl -X POST "http://0.0.0.0:8080/embed_jina" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is another sample sentence."}'
  ```

### 3. `/`

- **Description**: Root endpoint to check if the server is running.
- **Method**: `GET`
- **Response**:
  ```json
  {
    "message": "Welcome to the DuckDB Quackformers API!"
  }
  ```

- **Example `curl` Command**:
  ```bash
  curl "http://0.0.0.0:8080/"
  ```

## Notes

- Ensure that the Quackformers extension is properly installed and loaded in DuckDB.
- The server uses the `allow_unsigned_extensions` configuration to enable unsigned extensions.
- For local testing, you can modify the `server.py` script to load the extension from a local path.

## Troubleshooting

- If you encounter issues with the Quackformers extension, ensure it is built and installed correctly.
- Check that the required Python dependencies are installed.
- Verify that the server is running on the correct port (`8080` by default).