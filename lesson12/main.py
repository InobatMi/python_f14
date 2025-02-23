

import threading
import time
import concurrent.futures


def test(ID):
    print(f'Thread{ID} has started!')
    time.sleep(2)
    print(f'Thread{ID} has finished!')
    

x = threading.Thread(target=test, args=(1,))


# print('Main Thread has started')
# ls = []
# for i in range(3):
#     th = threading.Thread(target=test, args=(i,))
#     th.start()
#     ls.append(th)
# for thr in ls:
#     thr.join()
# print('Main Thread has finished')

print('Main Thread has started')

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(test,range(3))

print('Main Thread has finished')