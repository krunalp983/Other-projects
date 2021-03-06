Import time
Import RPi.GPIO as GPIO
Import requests
IFTTTmakersecretkey="dezCo1FXEOTrpKPNt3Hqs"
IFTTTfiredetectionURL="https://maker.ifttt.com/trigger/fire detection/with/key/dezCo1FXEOTrpKPNt3Hqs"


Import Adafruit_ADS1x15.ADS115()
from time impoer sleep
GPIO.setwarnings(False)

adc = Adafruit_ADS1X15.ADS1115()

GPIO:setmode(GPIO.BCM)
GreenLEDpin=20
RedLEDpin=21
Buzzer=18
Button=10

GPIO.setup(GreenLEDpin,GPIO.OUT)
GPIO.setup(RedLEDpin,GPIO.OUT)
GPIO.setup(Buzzer,GPIO.OUT)
GPIO.setup(GreenLEDpin,False)
GPIO.setup(RedLEDpin,False)
GPIO.setup(Button,GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

GAIN=1

Print('Reading ADS1X15 Values, Press Ctrl-C to quit....')

GPIO.setup(GreenLEDpin,False)
GPIO.setup(RedLEDpin,False)
while True:
 
     Value=[0]*2
     for i in range(2):
         Value[i] = adc.read_adc(i, gain=GAIN)
         
         FLAME=[0]
         GAS=[1]
     print("FLAME",Value[0],"GAS"[1])
     
     if (FLAME > 20000) and (1800<GAS<3000)
         print("GreenLEDpin will on")
         GPIO.output(GreenLEDpin,True)
         sleep(1)
         GPIO.output(RedLEDpin,False)
         GPIO.output(Buzzer,False)
         
     else:
         GPIO.output(GreenLEDpin,False)
    
     if FLAME < 20000 :
        print("REDLEDpin will on")
        GPIO.output(GreenLEDpin,False)
        sleep(1)
        GPIO.output(RedLEDpin,True)
        sleep(1)
        GPIO.output(Buzzer,True)
        sleep(1)
        r=request.get(IFTTTfiredetectionURL)
    
    else:
        GPIO.output(RedLEDpin,False)
        
     if GAS > 4000 :
        print("REDLEDpin will on")
        GPIO.output(GreenLEDpin,False)
        sleep(1)
        GPIO.output(RedLEDpin,True)
        sleep(1)
        GPIO.output(Buzzer,True)
        sleep(1)
        r=request.get(IFTTTfiredetectionURL)
        
     else:
        GPIO.output(RedLEDpin,False)
        
        
        
         
     
