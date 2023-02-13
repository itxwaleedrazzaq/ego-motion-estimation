// Steering position 
int right = 767;   
int left = 290;
int mid = 519;


int potPin = A0;
int ENA = 4;
int IN1 = 5;
int IN2 = 6;
int IN3 = 7;
int IN4 = 8;
int ENB = 9;
char chr;
char repeat = '0';
int incomming;
float set;

//---------------------------Function of Motor to move Forward ------------------------------//
void Forward()
{
  digitalWrite(IN3,HIGH);
  digitalWrite(IN4,LOW);
}

//---------------------------Function of Motor to move Backward ------------------------------//
void back()
{
  digitalWrite(IN3,LOW);
  digitalWrite(IN4,HIGH);
}

//---------------------------Function of Motor to Stop the Car ------------------------------//
void Stop()
{
  digitalWrite(IN3,LOW);
  digitalWrite(IN4,LOW);
}

//---------------------------Function of Car to turn right and avoid obstacle ------------------------------//
void Right_Turn()
{
  Stop();
  control(right);  // Controllong
  delay(500);         //Delay
  Forward();
  delay(4000);
  Stop();
  control(left);   //Controlling
  delay(500);
  Forward();
  delay(4000);
  Stop();
  control(mid);         // 519 = Straight 
  delay(500);
  Forward();
  delay(3000);
  Stop();
}

//---------------------------Function Car to turn left and avoid obstacle ------------------------------//
void Left_Turn()
{
  Stop();
  control(left);  // Controllong
  delay(500); //Delay
  Forward();
  delay(5000);
  control(right);   //Controlling
  delay(500);
  Forward();
  delay(500);
  control(mid);         // 519 = Straight 
  delay(500);
  Forward();
  delay(3000);
  Stop();
}

//---------------------------Function of Car to go to straight position ------------------------------//

void Straight()
{
  Stop();
  control(mid);
  delay(500);
  Forward();
  delay(3000);
  
}

//---------------------------Function of Car to go to backward position ------------------------------//
void Backward()
{
  Stop();
  control(mid);
  delay(500);
  back();
  delay(3000);
  
}

//---------------------------Steering Controller ------------------------------//
void control(int(incomming))
{
   while(1==1)
  {
  set = analogRead(A0);

if (incomming != 519)
{
  if(set < incomming )
  {
    digitalWrite(IN1,LOW);
    digitalWrite(IN2,HIGH);
    if( analogRead(A0)== 295+10 || analogRead(A0)== 295-10)
    {
      break;
    }}
  if(set > incomming && incomming != 519)
  {
    digitalWrite(IN1,HIGH);
    digitalWrite(IN2,LOW);
    if(analogRead(A0) == 767+10 || analogRead(A0) == 767 -10)
    {
      break;
    }}}
  if(incomming == 519)
  {
    if (set < incomming)
    {
    digitalWrite(IN1,LOW);
    digitalWrite(IN2,HIGH);
    if(analogRead(A0) == 519+5 || analogRead(A0) == 519-5)
    {
      break;
    }}

    if (set > incomming)
    {
    digitalWrite(IN1,HIGH);
    digitalWrite(IN2,LOW);
    if(analogRead(A0) == 519+10 || analogRead(A0) == 519-10)
    {
      break;
    }}}}
    digitalWrite(IN1,LOW);
    digitalWrite(IN2,LOW);
  }

//------------------------------------------- Void Setups to set pins-------------------------------------//
void setup() {
  Serial.begin(9600);
  pinMode(ENA,OUTPUT);   // motor controller pins
  pinMode(IN1,OUTPUT);
  pinMode(IN2,OUTPUT);
  pinMode(ENB,OUTPUT);
  pinMode(IN3,OUTPUT);
  pinMode(IN4,OUTPUT);
  
  analogWrite(ENA,150);   // setting PWM for both motor and steering
  analogWrite(ENB,255);
  
  
  control(mid);  // move the steering to default condition
  Stop();     // Stop the car
}


//-------------------------------------------------Void_LOOP------------------------------------------------//
void loop() {

  while(Serial.available() == 0){}
  chr = Serial.read();

  if (chr == repeat)
  {}

  else
  {
   repeat = chr; 
  if (chr == 'S'){ control(mid); Stop();}
  else if  (chr == 'R') { Right_Turn(); }
  else if  (chr == 'L') { Left_Turn();}
  else if  (chr == 'B') { Backward();  }
  else  { Straight();}
  }}
  
 
