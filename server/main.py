from fastapi import FastAPI, WebSocket
import json

app = FastAPI()

RADXA = None

@app.websocket("/radxa")
async def radxa_ws(ws: WebSocket):
    global RADXA
    await ws.accept()
    RADXA = ws
    print("Radxa connected")

    try:
        while True:
            msg = await ws.receive_text()
            print("From Radxa:", msg)
    except:
        RADXA = None
        print("Radxa disconnected")


@app.websocket("/control")
async def control_ws(ws: WebSocket):
    await ws.accept()
    print("Control connected")

    while True:
        cmd = await ws.receive_text()
        print("Command:", cmd)

        if RADXA:
            await RADXA.send_text(cmd)