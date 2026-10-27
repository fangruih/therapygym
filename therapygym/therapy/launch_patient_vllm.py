from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import create_openai_api_app
import uvicorn
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)  # Changed to 0.3
    args = parser.parse_args()

    engine_args = AsyncEngineArgs(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=8192,
        quantization=None,
        enforce_eager=False,
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    app = create_openai_api_app(engine)
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main() 
