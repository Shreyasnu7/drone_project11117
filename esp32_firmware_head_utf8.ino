/**
 * ULTRA DRONE - ESP32 SENSOR HUB (UART EDITION)
 * ------------------------------
 * Role: 
 *  1. Read 4x VL53L1X ToF Sensors (Omnidirectional)
 *  2. Read MPU6050 Gyro
 *  3. Control Gimbal Servos & LEDs
 *  4. Send JSON Telemetry to Radxa via UART (TX0/RX0)
 * 
 * WIRING (Verified):
 *  - TX0 -> Radxa GPIOX_9
 *  - RX0 -> Radxa GPIOX_10
 *  - TOF XSHUT: D23, D26, D15, D5
 *  - GIMBAL: D25, D32
 *  - CAM: D4, D34
 *  - LED: D33
 *  - I2C: D21 (SDA), D22 (SCL)
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_VL53L1X.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_NeoPixel.h>
#include <ESP32Servo.h>
#include <ArduinoJson.h>

// --- PIN DEFS (USER AUDITED) ---
#define PIN_XSHUT1 23 // ToF 1
#define PIN_XSHUT2 26 // ToF 2
#define PIN_XSHUT3 15 // ToF 3
#define PIN_XSHUT4 5  // ToF 4

#define PIN_LED 33
#define PIN_GIM_PITCH 25 // Gimbal 1
#define PIN_GIM_YAW 32   // Gimbal 2
#define PIN_CAM_TRIG 4   // Ext Cam Pin 3
#define PIN_CAM_TRIG2 34 // Ext Cam Pin 4

// --- OBJECTS ---
// 4 Sensors (PCB Fixed Layout)
Adafruit_VL53L1X tof1 = Adafruit_VL53L1X(PIN_XSHUT1);
Adafruit_VL53L1X tof2 = Adafruit_VL53L1X(PIN_XSHUT2);
Adafruit_VL53L1X tof3 = Adafruit_VL53L1X(PIN_XSHUT3);
Adafruit_VL53L1X tof4 = Adafruit_VL53L1X(PIN_XSHUT4);

Adafruit_MPU6050 mpu;
Adafruit_NeoPixel strip(16, PIN_LED, NEO_GRB + NEO_KHZ800);
Servo gimPitch;
Servo gimYaw;

// --- STATE ---
unsigned long last_send = 0;

// Safe Init Helper
bool startToF(Adafruit_VL53L1X &sensor, int id, int limitPin, uint8_t newAddr) {
    // 1. Manually Enable the sensor
    digitalWrite(limitPin, HIGH);
    delay(10);
    
    // 2. Initialize
    if (!sensor.begin(newAddr)) {
        return false; // Silent fail to loop
    }
    
    // 3. Start Ranging
    sensor.startRanging();
    return true;
}

void setup() {
    // V99: SERIAL UART LINK TO RADXA
    Serial.begin(115200); 
    
    // 1. Reset All Sensors (XSHUT LOW)
    pinMode(PIN_XSHUT1, OUTPUT);
    pinMode(PIN_XSHUT2, OUTPUT);
    pinMode(PIN_XSHUT3, OUTPUT);
    pinMode(PIN_XSHUT4, OUTPUT);
    
    digitalWrite(PIN_XSHUT1, LOW);
    digitalWrite(PIN_XSHUT2, LOW);
    digitalWrite(PIN_XSHUT3, LOW);
    digitalWrite(PIN_XSHUT4, LOW);
    delay(100);

    // 2. Init Peripherals
    pinMode(PIN_CAM_TRIG, OUTPUT);
    pinMode(PIN_CAM_TRIG2, OUTPUT);
    
    gimPitch.attach(PIN_GIM_PITCH);
    gimYaw.attach(PIN_GIM_YAW);
    
    strip.begin();
    strip.setPixelColor(0, strip.Color(255, 165, 0)); // Orange (Boot)
    strip.show();
    
    // 3. Start I2C (User: D21/D22)
    Wire.begin(21, 22);
    delay(100);

    // 4. Sequential Start
    if(startToF(tof1, 1, PIN_XSHUT1, 0x30)) strip.setPixelColor(1, strip.Color(0,255,0));
    if(startToF(tof2, 2, PIN_XSHUT2, 0x31)) strip.setPixelColor(2, strip.Color(0,255,0));
    if(startToF(tof3, 3, PIN_XSHUT3, 0x32)) strip.setPixelColor(3, strip.Color(0,255,0));
    if(startToF(tof4, 4, PIN_XSHUT4, 0x33)) strip.setPixelColor(4, strip.Color(0,255,0));
    
    strip.show();

    // MPU6050
    mpu.begin();
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    
    // Ready
    strip.setPixelColor(0, strip.Color(0, 0, 255)); // Blue (Ready)
    strip.show();
}

void loop() {
    // 1. READ SENSORS
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    
    int16_t d1 = -1, d2 = -1, d3 = -1, d4 = -1;
    if (tof1.dataReady()) { d1 = tof1.distance(); tof1.clearInterrupt(); }
    if (tof2.dataReady()) { d2 = tof2.distance(); tof2.clearInterrupt(); }
    if (tof3.dataReady()) { d3 = tof3.distance(); tof3.clearInterrupt(); }
    if (tof4.dataReady()) { d4 = tof4.distance(); tof4.clearInterrupt(); }
    
    // 2. READ COMMANDS (FROM RADXA UART)
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, input);
        if (!error) {
            // LED Control
            if (doc.containsKey("led")) {
                const char* c = doc["led"];
                if(strcmp(c, "RED")==0) strip.fill(strip.Color(255,0,0));
                if(strcmp(c, "GRN")==0) strip.fill(strip.Color(0,255,0));
                strip.show();
            }
            // Gimbal Control
            if (doc.containsKey("gim")) {
                int p = doc["gim"][0];
                int y = doc["gim"][1];
                gimPitch.write(map(p, -90, 90, 0, 180));
                gimYaw.write(map(y, -90, 90, 0, 180));
            }
            // Cam Trigger
            if (doc.containsKey("cam")) {
                 int state = doc["cam"];
                 digitalWrite(PIN_CAM_TRIG, state);
            }
        }
    }
    
    // 3. SEND TELEMETRY (20Hz)
    if (millis() - last_send > 50) {
        StaticJsonDocument<300> telem;
        
        telem["tof_1"] = d1;
        telem["tof_2"] = d2;
        telem["tof_3"] = d3;
        telem["tof_4"] = d4;
        
        JsonArray gyr = telem.createNestedArray("gyro");
        gyr.add(g.gyro.x); gyr.add(g.gyro.y); gyr.add(g.gyro.z);
        
        // Serialize directly to Serial (UART)
        serializeJson(telem, Serial);
        Serial.println(); // Newline for parser
        
        last_send = millis();
    }
}
