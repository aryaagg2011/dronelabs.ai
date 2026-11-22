# dronelabs.ai

Scripts for a Jetson Orin Nano + OAK-D Lite + Pixhawk (Holybro X500) drone.

## Guides (read in order)

1. [Bench setup](<docs/setup/Jetson%20Orin%20Nano%20+%20OAK-D%20Lite%20Bench%20Setup%20Tutorial%20(1).pdf>)
2. [SSH from a PC](docs/setup/Jetson%20Orin%20Nano%20Controlling%20It%20from%20a%20PC%20over%20SSH.pdf)
3. [Testing the camera](docs/usage/Testing_the_OAK-D_Lite_Device_Check_Color_and_Depth.pdf) → `diagnostics/`
4. [Wi-Fi hotspot for the field](docs/setup/Jetson%20WiFi%20Hotspot%20Tutorial.pdf)
5. [Command guide](docs/usage/Find_Person_Object_Stop_Commands_No_Monitor.pdf) → `obstacle_avoidance/`, `target_finding/`, `tools/`

## Folders

- `diagnostics/` – camera checks and benchmarks
- `detection/` – object detection with distance
- `obstacle_avoidance/` – depth-based stop, including the PX4 flight test
- `target_finding/` – find a flat target or a person
- `tools/` – `remote_view.py` (watch any script in a browser)
