import time

start_time = time.time()
time.sleep(10)
target_time = time.time() - start_time
print('target_time:', target_time)

def convert_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)
      
print(convert_seconds(target_time))