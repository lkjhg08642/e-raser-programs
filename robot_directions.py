import asyncio
from bleak import BleakClient, BleakScanner

# Replace with your Feather's BLE name or address
TARGET_NAME = "Nano33IoT_UART"
UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"

currentPos = [0,0]
targetPos = [0,0]
thresh = 5 #js what i have rn 
turnLength = 200 #ms need to tune 

async def main():
	devices = await BleakScanner.discover()
	for d in devices:
		if d.name:
			print(d.name)
			if TARGET_NAME in d.name:
				print(f"Found target: {d.name} [{d.address}]")
				async with BleakClient(d.address) as client:
					def handle_rx(_, data):
						print("Received:", data.decode())
					await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
					
					async def turn_right():
						await client.write_gatt_char(UART_RX_CHAR_UUID, b'RIGHT\n')
						await asyncio.sleep(turnLength/1000)
						await client.write_gatt_char(UART_RX_CHAR_UUID, b'COAST\n')
						heading += 90
					
					async def turn_left():
						await client.write_gatt_char(UART_RX_CHAR_UUID, b'LEFT\n')
						await asyncio.sleep(turnLength/1000)
						await client.write_gatt_char(UART_RX_CHAR_UUID, b'COAST\n')
						heading -= 90
					
					async def turn_to_heading(targetHeading):
						if (targetHeading - heading) == 90:
							turn_right()
						else if (targetHeading - heading) == -90: 
							turn_left()
						else if (targetHeading - heading) == 180 or (targetHeading - heading) == -180:
							turn_right()
							turn_right()
						else if (targetHeading - heading) == -270: 
							turn_right()
						else if (targetHeading - heading) == 270: 
							turn_left()
						else if (targetHeading - heading) == 0:
							pass 
					
					'''await client.write_gatt_char(UART_RX_CHAR_UUID, b'F\n')
					await asyncio.sleep(5)
					await client.write_gatt_char(UART_RX_CHAR_UUID, b'BRAKE\n')'''
					
					while True:
						await client.write_gatt_char(UART_RX_CHAR_UUID, b'F\n')
						
						#x movement 
						if (currentPos[0] > targetPos[0]):
							turn_to_heading(270)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'F\n')
							while currentPos[0] > targetPos[0] + thresh:
								await asyncio.sleep(0.1)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'BRAKE\n')
							
						else if (currentPos[0] < targetPos[0]):
							turn_to_heading(90)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'F\n')
							while currentPos[0] < targetPos[0] - thresh:
								await asyncio.sleep(0.1)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'BRAKE\n')
							
						#y movement
						if (currentPos[1] > targetPos[1]):
							turn_to_heading(0)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'F\n')
							while currentPos[1] > targetPos[1] + thresh:
								await asyncio.sleep(0.1)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'BRAKE\n')
						
						else if (currentPos[1] < targetPos[1]):
							turn_to_heading(180)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'F\n')
							while currentPos[1] < targetPos[1] - thresh:
								await asyncio.sleep(0.1)
							await client.write_gatt_char(UART_RX_CHAR_UUID, b'BRAKE\n')
						
					await asyncio.sleep(10)  # Keep connection alive for 10 seconds
					await client.stop_notify(UART_TX_CHAR_UUID)
				break
	else:
		print("Target not found.")

asyncio.run(main())
