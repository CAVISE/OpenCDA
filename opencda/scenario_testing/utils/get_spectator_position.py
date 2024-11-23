import carla  
import random  
  
client = carla.Client('localhost', 2000)  
world = client.get_world()  
  
spectator = world.get_spectator()  
print(spectator.get_transform())
