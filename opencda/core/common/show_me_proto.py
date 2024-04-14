import time
import json
from google.protobuf.json_format import MessageToDict
from opencda.core.common.protobuf_structure_pb2 import Cav  # Замените на имя вашего файла protobuf и соответствующего сообщения

def read_and_write_proto_file(proto_file, output_file):
    # Создание экземпляра сообщения для хранения данных из файла protobuf
    message = Cav()

    # Чтение содержимого файла protobuf
    with open(proto_file, "rb") as f:
        message.ParseFromString(f.read())
        print("Прочитано содержимое файла protobuf:")
        print(message)  

    # Преобразование сообщения в словарь JSON
    message_dict = MessageToDict(message)

    # Проверка наличия данных перед записью
    if message_dict:
        # Запись словаря в файл JSON
        with open(output_file, "w") as output_f:
            json.dump(message_dict, output_f, indent=4)
            output_f.flush()  # Принудительный сброс буфера записи
            print("Данные успешно записаны в файл JSON:", output_file)  
    else:
        print("Ошибка: Нет данных для записи в файл JSON")




class ProtoFileMonitor:
    def __init__(self, proto_file, output_file):
        self.proto_file = proto_file
        self.output_file = output_file

    def monitor(self):
        # Бесконечный цикл мониторинга файла protobuf и записи его содержимого в файл JSON
        while True:
            read_and_write_proto_file(self.proto_file, self.output_file)
            time.sleep(1)  # Пауза перед следующей проверкой

if __name__ == "__main__":
    proto_file = 'output_file.proto'
    output_file = "output.json"
    read_and_write_proto_file(proto_file, output_file)
