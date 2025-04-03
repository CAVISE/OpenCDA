import carla  

client = carla.Client('carla', 2000)  
world = client.get_world()  

spectator = world.get_spectator()  
print(spectator.get_transform())
