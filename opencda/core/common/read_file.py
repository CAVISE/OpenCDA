import protobuf_structure_pb2  # Import the generated Python code from your .proto file

def read_proto_file(filename):
    # Create a new Cav message
    cav_message = protobuf_structure_pb2.Cav()

    # Read the serialized data from the file
    with open(filename, 'rb') as f:
        serialized_data = f.read()

    # Parse the serialized data into the Cav message
    cav_message.ParseFromString(serialized_data)

    # Print the information
    print("vid:", cav_message.vid)
    print("ego_spd:", cav_message.ego_spd)
    print("ego_pos:", cav_message.ego_pos)
    print("nearby_vids:", cav_message.nearby_vids)
    print("vehicles:", cav_message.vehicles.cav_pos)
    print("traffic_lights:", cav_message.traffic_lights.tf_pos)
    print("static_objects:", cav_message.static_objects.obj_pos)
    print("who:", cav_message.who_received.received_vids)

# Example usage:
if __name__ == "__main__":
    read_proto_file('output_file.proto')
 