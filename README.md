# WindMesh 🔥

**Real-time wind field visualization and wildfire spread forecasting for incident commanders.**

WindMesh is a physics-based fire spread simulator that lets you place up to 7 configurable wind sensor nodes on any location in the world, set wind speed and direction at each one, and watch a live fire perimeter forecast update in real time on an interactive Google Maps interface. Built at a hackathon in one week for Diana — and every incident commander like her.

---

## The Problem

The #1 documented precursor to wildland firefighter deaths is an unexpected wind shift.

In 2013, 19 members of the Granite Mountain Hotshots were killed on the Yarnell Hill Fire when a thunderstorm outflow boundary caused a sudden change in wind direction, sending the fire racing south three times faster than it had been moving. They had a radio and a paper map. The NWCG's official fatality investigation framework lists wind shifts as a core common denominator across decades of tragedy fires.

Current tools fail in two specific ways:
- **Regional forecasts are too coarse.** Wind varies at the scale of tens of meters in complex terrain — valleys, ridgelines, canyons — far below what any weather model resolves.
- **Fixed stations go down when they're needed most.** In the 2007 Santa Ana fires, 8 of 15 automated weather stations had to be dropped from analysis because the fire and wind disrupted them.

WindMesh is built around the idea that cheap, distributed, throwable sensor nodes — placed by retreating crews — provide redundant, hyperlocal wind data exactly where the fire is.

---

## What It Does

- **7 configurable sensor nodes** placeable anywhere on a Google Maps interface
- **Per-node wind speed and direction** — set independently at each node to simulate real spatial wind variation
- **Simplified Rothermel fire spread model** — physics-based spread rate calculation driven by live wind inputs
- **Huygens ellipse perimeter** — fire perimeter projected forward in time, oriented along the wind field
- **Uncertainty halo** — confidence envelope that widens with distance from sensor coverage
- **Crew position marker** — place a crew on the map and see time-to-reach update in real time
- **Fully customizable location** — simulate any fire, anywhere in the world
- **Live sensor mode** — plug in a real ESP32 + BME280 node over USB serial for actual pressure and temperature readings

---

## Demo

> Point a fan at the sensor node. Watch the fire perimeter grow and shift direction in real time.

![WindMesh demo](docs/demo.gif)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend map | Leaflet.js on OpenStreetMap tiles |
| Backend | Python (FastAPI + uvicorn) |
| Real-time comms | WebSocket |
| Fire spread model | Simplified Rothermel (NFFL Fuel Model 4) |
| Wind field | Open-Meteo API + per-node manual input |
| Hardware node | ESP32 DevKit + BME280 over USB serial |
| Sensor firmware | Arduino (C++) |

---

## Physics

### Rothermel Fire Spread Model

WindMesh uses a simplified implementation of the Rothermel (1972) fire spread equation, the standard model used by the U.S. Forest Service for operational fire behavior prediction:

```
R = (I_R · ξ · (1 + φ_w + φ_s)) / (ρ_b · ε · Q_ig)
```

Where:
- `R` — rate of spread (ft/min)
- `I_R` — reaction intensity (BTU/ft²/min)
- `ξ` — propagating flux ratio
- `φ_w` — wind factor (dominant variable in flat terrain)
- `φ_s` — slope factor (set to 0 in current demo)
- `ρ_b` — fuel bed bulk density (lb/ft³)
- `ε` — effective heating number
- `Q_ig` — heat of preignition (BTU/lb)

Fuel constants are hardcoded to **NFFL Fuel Model 4 (Chaparral)** — representative of California and southeastern Australian fire terrain. Wind speed is the live variable that drives the perimeter in the demo.

### Fire Perimeter Geometry (Huygens Principle)

The perimeter is modeled as an ellipse oriented along the wind direction:
- **Major axis** (downwind): `R · t · 0.3048` meters
- **Minor axis** (crosswind): `major × 0.5`

This is the standard Huygens wavelet approximation used in FARSITE and similar operational fire perimeter tracking tools.

### Uncertainty Halo

The halo around the perimeter represents forecast confidence. It is computed as a Gaussian Process (GP) posterior variance field:
- Uncertainty is low near sensor nodes (well-observed)
- Uncertainty grows with distance from the nearest node
- Halo width = 20% of the major axis radius, modulated by node density

In the multi-node version, a `sklearn GaussianProcessRegressor` interpolates the full wind field across the map and the halo reflects genuine spatial uncertainty between observation points.

---

## Hardware (Optional Live Mode)

For a live demo with real sensor data, you need:

| Component | Purpose | ~Cost |
|---|---|---|
| ESP32 DevKit v1 | Microcontroller, USB serial | $8 |
| BME280 breakout | Temperature, humidity, pressure | $8 |
| Breadboard + jumpers | Wiring | $5 |
| USB-A to Micro-USB | ESP32 to laptop | $4 |

**Wiring (I2C):**
```
BME280 VIN  →  ESP32 3.3V
BME280 GND  →  ESP32 GND
BME280 SCL  →  ESP32 GPIO 22
BME280 SDA  →  ESP32 GPIO 21
```

Flash `firmware/anemometer_node.ino` using Arduino IDE with the `Adafruit BME280` and `ArduinoJson` libraries installed.

The node transmits a JSON packet over USB serial at 115200 baud every 500ms:
```json
{
  "pressure": 1013.2,
  "temp": 22.4,
  "humidity": 38.1,
  "dp_dt": -0.012,
  "timestamp": 48291
}
```

A rapid pressure drop (`dp_dt < 0`) is used as a proxy for incoming wind — physically motivated by the relationship between pressure gradients and wind-generating atmospheric dynamics.

---

## Setup

### Prerequisites

- Python 3.9+
- Node with an active internet connection (for Open-Meteo wind API)

### Install

```bash
git clone https://github.com/yourusername/windmesh.git
cd windmesh
pip install -r requirements.txt
```

### Run (simulation mode — no hardware needed)

```bash
python backend.py
```

Open `http://localhost:8000` in your browser.

### Run (live sensor mode)

1. Flash the firmware to your ESP32
2. Connect via USB
3. Find your serial port:
   - **Mac/Linux:** `ls /dev/tty.*` — look for `ttyUSB0` or `tty.usbserial-*`
   - **Windows:** Check Device Manager → Ports → `COM3` or similar
4. Update `SERIAL_PORT` in `backend.py`
5. Run `python backend.py`

### Configuration

Key values to adjust in `backend.py`:

```python
SERIAL_PORT = "/dev/ttyUSB0"   # Your ESP32 port
DEMO_LAT    = 37.4419          # Map center latitude
DEMO_LON    = -122.1430        # Map center longitude
FIRE_ORIGIN = [37.4400, -122.1400]  # Fire start point
```

And in `firmware/anemometer_node.ino`:

```cpp
// Adjust if your BME280 uses address 0x77 instead of 0x76
if (!bme.begin(0x76)) { ... }

// Transmission interval in milliseconds
if (millis() - lastSend > 500) { ... }
```

---

## File Structure

```
windmesh/
├── backend.py              # FastAPI server, Rothermel model, WebSocket
├── requirements.txt
├── static/
│   └── index.html          # Leaflet map UI, WebSocket client
├── firmware/
│   └── anemometer_node.ino # ESP32 sensor firmware
├── docs/
│   └── demo.gif
└── README.md
```

---

## Roadmap

The current build is a proof of concept with one physical node and a simulated multi-node interface. The full architecture is designed to scale:

- [ ] **LoRa mesh radio** — replace USB serial with RFM95W point-to-point link (same JSON payload, same backend)
- [ ] **Multi-node mesh routing** — Meshtastic or custom flood-fill routing over 915 MHz
- [ ] **Real anemometer** — cup anemometer or ultrasonic wind sensor replacing pressure-gradient proxy
- [ ] **Ensemble Kalman Filter** — replace static wind inputs with a self-correcting ensemble that updates as new sensor packets arrive
- [ ] **Full kriging interpolation** — spatial wind field estimation between nodes using `sklearn GaussianProcessRegressor`
- [ ] **Fuel model selector** — parameterize for different vegetation types (grassland, chaparral, timber)
- [ ] **Slope integration** — add φ_s term from DEM elevation data
- [ ] **Crew GPS integration** — pull real crew positions from radio GPS rather than manual map placement
- [ ] **Mobile UI** — field-optimized layout for tablet use in incident command

---

## Motivation and Evidence

This was built for a hackathon theme: **Build for Someone.**

We built it for Diana — an incident commander managing crew positioning during an active fire. She has a radio, a paper map, and information that's already 20 minutes old. Wind conditions are changing. She has to decide right now whether to pull her crews back.

The evidence that her problem is real:
- Unexpected wind shifts are listed as a common denominator in **every major wildland firefighter fatality investigation** (NWCG)
- **400+ on-duty wildland firefighter deaths** occurred in the U.S. from 2000–2019 (CDC)
- The 2013 Yarnell Hill Fire — where 19 of 20 Granite Mountain Hotshots were killed by a sudden wind shift — was the deadliest U.S. firefighter disaster since 9/11
- Rate of Spread calculations are considered **highly inaccurate due to uncertainty in local wind data** (ScienceDirect, 2021)
- In the 2007 Santa Ana fires, **8 of 15 automated weather stations failed** during the event (arxiv.org)
- Australia's Black Summer (2019–20) burned 24 million hectares and caused ~$100B in damage, with wind the primary driver of uncontrollable fire behavior (CSIRO)

WindMesh doesn't replace incident meteorologists. It gives them — and Diana — ground-truth wind data where the fire actually is.

---

## Team

Built in one week at LexHack '26 - June 6-13 2026

By: Toshan Goswami, Ishaan Jagali, Atharva Ranjan

---

## References

- Rothermel, R.C. (1972). *A Mathematical Model for Predicting Fire Spread in Wildland Fuels.* USDA Forest Service Research Paper INT-115.
- NWCG. *Common Denominators of Fire Behavior on Tragedy Fires.* https://www.nwcg.gov/6mfs/weather-fire-behavior/common-denominators-of-fire-behavior-on-tragedy-fires
- CDC/NIOSH. *Wildland Fires — Firefighter Fatalities.* https://www.cdc.gov/niosh/firefighters/about/wildfires.html
- Finney, M.A. (1998). *FARSITE: Fire Area Simulator — Model Development and Evaluation.* USDA Forest Service RMRS-RP-4.
- Open-Meteo API. https://open-meteo.com

---

## License

MIT
