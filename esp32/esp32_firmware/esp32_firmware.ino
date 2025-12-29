void setup() {
  Serial.begin(115200);
}

void loop() {
  Serial.println("ESP32_ALIVE");
  delay(1000);
}