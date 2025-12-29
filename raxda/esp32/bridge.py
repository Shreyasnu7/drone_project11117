import serial

esp = serial.Serial('/dev/ttyUSB1', 115200)

def read_esp():
    if esp.in_waiting:
        return esp.readline().decode().strip()