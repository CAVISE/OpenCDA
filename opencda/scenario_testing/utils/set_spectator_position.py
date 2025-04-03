import carla  

client = carla.Client('carla', 2000)  
world = client.get_world()  

spectator = world.get_spectator()  

x, y, z = map(float, input().split(","))  
location = carla.Location(x=x, y=y, z=z)  
rotation = carla.Rotation(pitch=0, yaw=-180, roll=0)  
spectator.set_transform(carla.Transform(location, rotation))
