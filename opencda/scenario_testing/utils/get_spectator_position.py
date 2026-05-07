import carla

client = carla.Client("carla", 2000)
world = client.get_world()

spectator = world.get_spectator()
location = spectator.get_transform().location
rotation = spectator.get_transform().rotation
print(f"Location: {location.x:.2f}, {location.y:.2f}, {location.z:.2f},")
print(f"Rotation: {rotation.pitch:.2f}, {rotation.yaw:.2f}, {rotation.roll:.2f}")
