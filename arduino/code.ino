// Define pins for ultrasonic sensor
const int trigPin = 9;
const int echoPin = 10;


// Define pin for buzzer
const int buzzerPin = 11;


// Variables for duration and distance
long duration;
int distance;


// Variables for controlling buzzer sound
int buzzerFrequency = 0;
int buzzerAmplitude = 0;


void setup() {
 // Initialize Serial Monitor
 Serial.begin(9600);


 // Define pin modes
 pinMode(trigPin, OUTPUT);
 pinMode(echoPin, INPUT);
 pinMode(buzzerPin, OUTPUT);
}


void loop() {
 // Clear the trigPin
 digitalWrite(trigPin, LOW);
 delayMicroseconds(2);


 // Send a 10us pulse to trigger
 digitalWrite(trigPin, HIGH);
 delayMicroseconds(10);
 digitalWrite(trigPin, LOW);


 // Read the duration of the pulse
 duration = pulseIn(echoPin, HIGH);


 // Calculate the distance (in cm)
 distance = duration * 0.034 / 2;


 // Print distance on Serial Monitor
 Serial.print("Distance: ");
 Serial.print(distance);
 Serial.println(" cm");


 // Adjust buzzer sound based on distance
 if (distance < 50) {
   // Calculate frequency and amplitude based on distance
   buzzerFrequency = map(distance, 0, 50, 1000, 2000);
   buzzerAmplitude = map(distance, 0, 50, 255, 0);
 } else {
   // If distance is greater than 50 cm, turn off the buzzer
   buzzerFrequency = 0;
   buzzerAmplitude = 0;
 }


 // Set buzzer frequency and amplitude
 tone(buzzerPin, buzzerFrequency, buzzerAmplitude);


 // Wait a short while before next reading
 delay(100);
}


