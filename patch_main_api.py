import re
with open("main.py", "r") as f:
    content = f.read()

# Replace the __main__ block
new_main = """
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI, Request
    
    app = FastAPI(title="AHRAS Telemetry API")
    pipeline = AHRASPipeline()
    
    @app.post("/api/v1/telemetry")
    async def ingest_telemetry(request: Request):
        try:
            event = await request.json()
            # Extract entity key (e.g. source IP)
            entity_key = event.get("src_endpoint", {}).get("ip", "unknown_entity")
            
            # Process through the unified pipeline
            result = pipeline.process_telemetry(entity_key=entity_key, event=event)
            
            return {
                "status": "processed",
                "entity": entity_key,
                "risk_score": result.risk_score,
                "action_taken": result.autonomy_decision
            }
        except Exception as e:
            log.error(f"Error processing telemetry: {e}")
            return {"status": "error", "message": str(e)}

    @app.get("/health")
    def health_check():
        return {"status": "AHRAS Engine Online"}

    log.info("Starting AHRAS FastAPI Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

content = re.sub(r'if __name__ == "__main__":.*', new_main.strip('\n'), content, flags=re.DOTALL)

with open("main.py", "w") as f:
    f.write(content)
