#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import socket
import threading
import queue
import time
import struct
import uuid
import numpy

import schedule
import hurst
import matplotlib.pyplot as pyplot
from matplotlib.table import table

from keras.models import Model
from keras.layers import Input, LSTM, Dense

__DEBUG__ = False
__DEBUG_TIME__ = 10
__RECOGNITION_THRESHOLD__ = 0.6

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class NeuralNetwork(threading.Thread, metaclass = Singleton):
    def __init__(self, dynamic = False, debug = False, *args, **kwargs):
        super(NeuralNetwork, self).__init__(*args, **kwargs)
        self.__BATCH_SIZE__ = 20
        self.__EPOCHS__ = 60
        self.__ACTIVATION__ = 'softmax'
        self.__OPTIMIZER__ = 'rmsprop'
        self.__LOSS__ = 'categorical_crossentropy'
        self.__LATENT_DIM__ = 128
        self.__VALIDATION_SPLIT__ = 0.2
        self.__MIN_NUM_SAMPLES__ = 100
        self.__MAX_TIME_SERIES__ = 100
        self.__UPDATE_FREQUENCY__ = 1
        self.__wait_event, self.__stop_event, self.__train_event, self.__buffer_event = threading.Event(), threading.Event(), threading.Event(), threading.Event()
        self.__wait_condition = threading.Condition(threading.Lock())
        self.__update_datasets_thread = threading.Thread(target = self.update_datasets)
        self.__path = get_path_cmd() + folder_name() + os.path.join(' ',' ')[1]
        self.__encoder = LSTM(self.__LATENT_DIM__, return_state = True)
        self.__decoder_lstm = LSTM(self.__LATENT_DIM__, return_sequences = True, return_state = True)
        self.__input_data, self.__output_data, self.__buffer = [], [], []
        self.__datafiles_names, self.__input_values, self.__output_values = set(), set(), set()
        self.__dynamic = dynamic
        self.__debug = debug
        self.__wait_event.set()
        if(self.__dynamic): self.__dataset = queue.Queue(self.__NUM_SAMPLES__)
        else: 
            self.__dataset = queue.Queue()
            Analyzer().pause()

    def update_datasets(self):
        def _():
            with threading.Lock():
                datafiles_names = {_ for _ in os.listdir(self.__path) if os.path.isfile(os.path.join(self.__path, _))} 
                if (len(datafiles_names - self.__datafiles_names) > 0):
                    self.clear_buffer()
                    for _ in (datafiles_names):
                        with open(self.__path + _, 'r') as file:
                            data = file.read().split('\n')
                            data.pop()
                            self.__dataset.put_nowait(data)
                    self.__datafiles_names = datafiles_names
                    self.__buffer_event.clear()
                    self.__train_event.set()
                    Analyzer().pause()
                    
        thread = threading.Thread(target = _)
        thread.start()
        thread.join()
        schedule.every(self.__UPDATE_FREQUENCY__).seconds.do(_)
        
        while not self.__stop_event.is_set(): 
            if(self.__buffer_event.is_set() and self.__debug):
                if(time.time() > self.__stop_time):
                    Analyzer().stop()
                    Analyzer().join()
                    intervals = []
                    for interval in self.__buffer:
                        sub_interval = []
                        interval = [_ for _ in interval if _ != '\n']
                        for _ in [interval[0 + i:4 + i] for i in range(0, len(interval), 4)]: sub_interval.append(''.join(_))
                        intervals.append([float(_) for _ in sub_interval])         
                    if(len(intervals) > 0):
                        intervals = list(map(list, zip(*intervals)))
                        index = numpy.arange(len(max(intervals, key = len)))
                        width = 0.7 / len(intervals)
                        colors = pyplot.cm.BuPu(numpy.linspace(0.4, 1, len(intervals)))[::-1]
                        labels = ['Sub-interval ' + str(_) for _ in range(len(intervals))]
                        for _, v in enumerate(intervals):
                            pyplot.bar(index + width * _, v, width, alpha = 0.85, color = colors[_], label = labels[_])
                        pyplot.xlabel('Intervals')
                        pyplot.ylabel('Hurst exponent')
                        pyplot.title('Traffic self-similarity')
                        pyplot.yticks(numpy.arange(0, 1.1, 0.1))
                        pyplot.xticks(index)
                        pyplot.legend()
                        pyplot.tight_layout()
                                            
                        ax = pyplot.subplots()[1]
                        ax.set_yticklabels([])
                        ax.set_xticklabels([])
                        ax.axis('off')
                        stats_table = table(ax, cellText = intervals, rowLabels = labels, 
                                        colLabels = ['Interval ' + str(_) for _ in range(len(max(intervals, key = len)))],
                                        bbox = [0.25, 0, 0.75, 0.6])
                        stats_table.set_fontsize(12)
                        pyplot.show()
                    
                    self.__debug = False
            schedule.run_pending()
            time.sleep(self.__UPDATE_FREQUENCY__)                                                                            

    def run(self, *args, **kwargs):
        super(NeuralNetwork, self).run(*args, **kwargs)
        if(not self.__dynamic): self.__update_datasets_thread.start()
                           
        def initialize():
            self.__input_values, self.__output_values = sorted(list(self.__input_values)), sorted(list(self.__output_values))
            self.__input_index = dict([(__, _) for _, __ in enumerate(self.__input_values)])
            self.__output_index = dict([(__, _) for _, __ in enumerate(self.__output_values)])
            self.__reverse_input_index = dict((_, __) for __, _ in self.__input_index.items())
            self.__reverse_output_index = dict((_, __) for __, _ in self.__output_index.items())
            self.__max_enc_data_length = max([len(_) for _ in self.__input_data])
            self.__max_dec_data_length = max([len(_) for _ in self.__output_data])
            self.__len_input_data, self.__len_output_data = len(self.__input_data), len(self.__output_data)
            self.__len_output_values = len(self.__input_values)
            
            self.__enc_input_data = numpy.zeros((self.__len_input_data, self.__max_enc_data_length, self.__MAX_TIME_SERIES__), dtype='float32')
            self.__dec_input_data = numpy.zeros((self.__len_input_data, self.__max_dec_data_length, self.__MAX_TIME_SERIES__), dtype='float32')
            self.__dec_target_data = numpy.zeros((self.__len_input_data, self.__max_dec_data_length, self.__MAX_TIME_SERIES__), dtype='float32')
                
            for _, (input_row, output_row) in enumerate(zip(self.__input_data, self.__output_data)):
                for __, value in enumerate(input_row): self.__enc_input_data[_, __, self.__input_index[value]] = 1.
                for __, value in enumerate(output_row):
                    self.__dec_input_data[_, __, self.__output_index[value]] = 1.
                    if __ > 0: self.__dec_target_data[_, __ - 1, self.__output_index[value]] = 1.
                
        def normalize_data(data):
            for _ in data:
                time_series, res_series = _.split('\t')
                norm_res_series, norm_time_series = [], time_series.split(' ')
                norm_res_series.append('\t')
                for __ in res_series.split(' '): norm_res_series.append(__)
                norm_res_series.append('\n')
      
                self.__input_data.append(norm_time_series)
                self.__output_data.append(norm_res_series)
                self.__input_values.update(norm_time_series)
                self.__output_values.update(norm_res_series)
            
        while not self.__stop_event.is_set():
            with self.__wait_condition:
                while not self.__wait_event.is_set():
                    self.__wait_condition.wait()
                if(self.__train_event.is_set()):
                    try:
                        while 1: 
                            normalize_data(self.__dataset.get_nowait())
                    except queue.Empty:
                        initialize()
                        enc_inputs = Input(shape = (None, self.__MAX_TIME_SERIES__))
                        enc_states = self.__encoder(enc_inputs)[1:]
                        dec_inputs = Input(shape = (None, self.__MAX_TIME_SERIES__))
                        dec_outputs = self.__decoder_lstm(dec_inputs, initial_state = enc_states)[0]
                        dec_dense = Dense(self.__MAX_TIME_SERIES__, activation = self.__ACTIVATION__)
                        dec_outputs = dec_dense(dec_outputs)
                            
                        model = Model([enc_inputs, dec_inputs], dec_outputs)
                        model.compile(optimizer = self.__OPTIMIZER__, loss = self.__LOSS__)
                        model.fit([self.__enc_input_data, self.__dec_input_data], self.__dec_target_data, 
                                      batch_size = self.__BATCH_SIZE__, epochs = self.__EPOCHS__, validation_split = self.__VALIDATION_SPLIT__)
    
                        self.__enc_model = Model(enc_inputs, enc_states)
                        dec_states_inputs = [Input(shape = (self.__LATENT_DIM__,)), Input(shape = (self.__LATENT_DIM__,))]
                        dec_outputs, state_h, state_c = self.__decoder_lstm(dec_inputs, initial_state = dec_states_inputs)
                        dec_outputs = dec_dense(dec_outputs)
                        self.__dec_model = Model([dec_inputs] + dec_states_inputs, [dec_outputs] + [state_h, state_c])
                            
                        self.__train_event.clear()
                        self.__buffer_event.set()
                        if(not self.__dynamic): Analyzer().resume()
                        if(self.__debug): self.__stop_time = time.time() + __DEBUG_TIME__
                if(self.__buffer_event.is_set()):
                    try:
                        data = self.__dataset.get_nowait()
                        norm_data = numpy.zeros((1, len(data), self.__MAX_TIME_SERIES__), dtype='float32')
                        input_index = dict([(__, _) for _, __ in enumerate(sorted(list(set(data))))])
                        for _, value in enumerate(data): 
                            norm_data[0, _, input_index[value]] = 1.
                        result_value = self.__enc_model.predict(norm_data)
                        output_data = numpy.zeros((1, 1, self.__MAX_TIME_SERIES__))
                        output_data[0, 0, self.__output_index['\t']] = 1.
                        result_series = ''
                        while 1:
                            output_values, h, c = self.__dec_model.predict([output_data] + result_value)
                            output_index = numpy.argmax(output_values[0, -1, :])
                            sampled_value = output_index
                            result_series += self.__reverse_output_index[sampled_value]        
                            output_data = numpy.zeros((1, 1, self.__MAX_TIME_SERIES__))
                            output_data[0, 0, output_index] = 1.
                            result_value = [h, c]  
                            if (len(result_series) > self.__max_dec_data_length): break
                        self.__buffer.append(result_series)
                        intervals = []
                        for interval in self.__buffer:
                            sub_interval = []
                            interval = [_ for _ in interval if _ != '\n']
                            for _ in [interval[0 + i:4 + i] for i in range(0, len(interval), 4)]: sub_interval.append(''.join(_))
                            intervals.append([float(_) for _ in sub_interval])         
                        if(len(intervals) > 1):
                            for _ in range(len(intervals)):
                                if(round(numpy.mean(intervals[_ - 1]), 2) == round(numpy.mean(intervals[_]), 2)):
                                    probability = 0
                                    for v1, v2 in zip(intervals[_ - 1], intervals[_]):
                                        if(v1 == v2): probability += 1
                                    probability =  round(probability * len(intervals[0]) / 100, 2)          
                                    if(probability > __RECOGNITION_THRESHOLD__):
                                        result_cmd()
                                        break
                    except queue.Empty: continue
                    except KeyError: continue
	
    def clear_buffer(self):
        try:
            while 1: self.__dataset.get_nowait()
        except queue.Empty: pass
        
    def add_dataset(self, data):
        try:
            self.__dataset.put_nowait(data)
        except queue.Full: self.__train_event.set()
        
    def get_results(self):
        data, norm_data, intervals = [], [], []
        for interval in self.__buffer:
            sub_interval = []
            interval = [v for v in interval if v != '\n']
            for _ in [interval[0 + i:4 + i] for i in range(0, len(interval), 4)]: sub_interval.append(''.join(_))
            intervals.append(sub_interval)
            for _ in intervals:
                num_sub_intervals, len_sub_intervals, result = 1, 1, []
                while(len_sub_intervals < len(_)):
                    num_sub_intervals += 1
                    len_sub_intervals += num_sub_intervals
                    for i in range(num_sub_intervals): 
                        index = 0
                        sub_interval = []
                        for k in range(i):
                            index += k
                            sub_value = _[index:index+k+1] 
                            norm_sub_value = [float(_) for _ in sub_value]
                            sub_interval.append(truncate(sum(norm_sub_value) / float(len(norm_sub_value)), 2))  
                        if(len(sub_interval) > 0): result.append(sub_interval) 
                    data.append(result)
                for interval in data:
                    norm_interval = []
                    for value in interval:
                        norm_value = [float(_) for _ in value]
                        norm_interval.append(float(truncate(sum(norm_value) / float(len(norm_value)), 2)))  
                    else: 
                        norm_data.append(norm_interval)
        return norm_data
            
    def resume(self):
        if(not self.__wait_event.is_set()):
            self.__wait_event.set()
            self.__wait_condition.notify()
            self.__wait_condition.release()
       
    def pause(self):
        if(self.__wait_event.is_set()):
            self.__wait_event.clear()
            self.__wait_condition.acquire()
    
    def stop(self):
        self.__stop_event.set() 
        if(not self.__dynamic): self.__update_datasets_thread.join()
        
class Analyzer(threading.Thread, metaclass = Singleton): 
    def __init__(self, collect = False, *args, **kwargs):
        super(Analyzer, self).__init__(*args, **kwargs)
        if(sys.platform.startswith('linux')):
            self.__socket = socket.socket(socket.PF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0800))
        elif(sys.platform.startswith('win32')):
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_IP)
            self.__socket.bind((socket.gethostbyname(socket.gethostname()), 0))
            self.__socket.setsockopt(socket.IPPROTO_IP, socket.IP_HDRINCL, 1)
            self.__socket.ioctl(socket.SIO_RCVALL, socket.RCVALL_ON)
        else: 
            print('Unsupported operating system')
            exit_cmd() 
        self.__SOCKET_BUFFER_SIZE__ = 65565
        self.__SCALE__ = 20
        self.__TIME_INTERVAL__ = 1000
        self.__file_name = get_path_cmd() + folder_name() + os.path.join(' ',' ')[1] + str(uuid.uuid4()) + '.txt'
        self.__stop_event, self.__wait_event, self.__queue_event = threading.Event(), threading.Event(), threading.Event()
        self.__buffer_wait_condition, self.__wait_condition = threading.Condition(threading.Lock()), threading.Condition(threading.Lock())
        self.__reduce_buffer_thread = threading.Thread(target = self.reduce_buffer, args = [collect], daemon = True)
        self.__buffer = queue.Queue()
        self.__sub_intervals, self.__norm_sub_intervals = [], []
        self.__wait_event.set()
        self.__queue_event.set()
         
    def reduce_buffer(self, collect):
        def get_hurst_exponent(time_series):
            Hurst_exponent = float(truncate(hurst.compute_Hc(time_series, kind = 'change', simplified = True)[0], 2))
            if(Hurst_exponent > 1): Hurst_exponent = 1.00
            elif(Hurst_exponent == 0): Hurst_exponent = 0.01
            return Hurst_exponent
        
        while not self.__stop_event.is_set():
            with self.__buffer_wait_condition:
                while not self.__wait_event.is_set():
                    self.__buffer_wait_condition.wait()
                time.sleep(self.__TIME_INTERVAL__ / 1000.0)
                self.__queue_event.clear()
                try:
                    data = []
                    while 1: data.append(self.__buffer.get_nowait()) 
                except queue.Empty: 
                   if(len(data) > 0):
                       sub_interval = [data.count(_) for _ in list(range(data[0], data[0] + self.__TIME_INTERVAL__))]
                       norm_sub_interval = list(numpy.add.reduceat(sub_interval, numpy.arange(0, len(sub_interval), self.__SCALE__)))
                       while norm_sub_interval[-1] == 0: norm_sub_interval.pop(-1)
                       self.__sub_intervals.append(sub_interval)
                       self.__norm_sub_intervals.append(norm_sub_interval)
                       norm_interval = []
                       for _ in self.__norm_sub_intervals: norm_interval.extend(_)
                       for _, value in enumerate(norm_interval): norm_interval[_] = str(value)
                       result = ' '.join(norm_interval) + '\t'
                       
                       for end in range(len(self.__sub_intervals) - 1, -1, -1):
                           target_interval = []
                           for start in range(len(self.__sub_intervals)):
                               target_sub_intervals = self.__sub_intervals[start:end]
                               if(len(self.__sub_intervals) == 1): target_sub_intervals.append(self.__sub_intervals[0])
                               if(len(target_sub_intervals) > 0):
                                   for _ in target_sub_intervals: target_interval.extend(_)
                                   result += str(get_hurst_exponent(target_interval)) + ' '
                       result = result[:-1]
                       result += '\n'   
                       if(collect): 
                           if(not os.path.isfile(self.__file_name)): self.__file = open(self.__file_name, 'w')
                           try:
                               self.__file.write(result)
                           except ValueError: pass
                       elif(NeuralNetwork().is_alive()): NeuralNetwork().add_dataset(result)
                self.__queue_event.set()
    
    def run(self, *args, **kwargs):
        super(Analyzer, self).run(*args, **kwargs)
        self.__reduce_buffer_thread.start()
        
        while not self.__stop_event.is_set():
            with self.__wait_condition:
                while not self.__wait_event.is_set():
                    self.__wait_condition.wait() 
                try:    
                    raw_data = self.__socket.recvfrom(self.__SOCKET_BUFFER_SIZE__)[0]
                    if(self.__queue_event.is_set()):
                        data = struct.unpack('!BBHHHBBH4s4s', raw_data[14:34])
                        if(socket.inet_ntoa(data[8]) == '127.0.0.1'): data = None
                        else: data = data[2], socket.inet_ntoa(data[8]), round(time.time() * 1000)
                        if(data is not None): self.__buffer.put_nowait(data[2])
                except ValueError: continue
    
    def get_results(self):
        return self.__ip_addresses
          
    def resume(self):
        if(not self.__wait_event.is_set()):
            self.__wait_event.set()
            self.__wait_condition.notify()
            self.__buffer_wait_condition.notify()
            self.__wait_condition.release()
            self.__buffer_wait_condition.release()
      
    def pause(self):
        if(self.__wait_event.is_set()):
            self.__wait_event.clear()
            self.__wait_condition.acquire()
            self.__buffer_wait_condition.acquire()
        
    def stop(self):
        self.__stop_event.set()
        if(sys.platform.startswith('win32')): self.__socket.ioctl(socket.SIO_RCVALL, socket.RCVALL_OFF)
        if(os.path.isfile(self.__file_name)): self.__file.close()
        
def truncate(float_num, trunc_num):
    str_num = '{}'.format(float_num)
    before_sep, sep, after_sep = str_num.partition('.')
    return '.'.join([before_sep, (after_sep + '0' * trunc_num)[:trunc_num]])        
        
def start_cmd():
    try:
        NeuralNetwork(debug = __DEBUG__).start()
        Analyzer().start()
    except KeyboardInterrupt: exit_cmd() 

def folder_name():
    return 'traffic'

def result_cmd():
    print("Similar traffic detected..")
    input("\tPress Enter to continue...")
    NeuralNetwork().clear_buffer()
    
def exit_cmd():
    analyzer = Analyzer()
    analyzer.stop()
    analyzer.join()
    neural_network = NeuralNetwork()
    if(neural_network.is_alive()):
        neural_network.stop()
        neural_network.join()
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
  
def set_path_cmd(default = False):
    if(len(sys.argv) != 3 and not default): raise FileNotFoundError
    with open(os.path.realpath(__file__), 'r+') as file:
        if(default): 
            if(not file.readlines()[-1][0] == '#'): 
               file.write('\n#' + os.getcwd() + os.path.join(' ',' ')[1])
        else: 
            file.seek(0, os.SEEK_END)
            pos = file.tell() - 1
            while pos > 0 and file.read(1) != '\n':
                pos -= 1
                file.seek(pos, os.SEEK_SET)
            if pos > 0:
                file.seek(pos, os.SEEK_SET)
                file.truncate()
            file.write('\n#' + sys.argv[2])  
                         
def get_path_cmd():
    with open(os.path.realpath(__file__), 'r+') as file: 
        fpath = file.readlines()[-1][1:]   
        if(len(sys.argv) == 2): print(fpath)
        else: return fpath  
    
def collect_cmd():
    if(len(sys.argv) != 3): raise ValueError
    int(sys.argv[2])
    if(not os.path.exists(get_path_cmd() + folder_name())): os.makedirs(get_path_cmd() + folder_name())
    try:
        analyzer = Analyzer(collect = True)
        analyzer.start()
        analyzer.join(int(sys.argv[2]))
        analyzer.stop()
        analyzer.join()
    except KeyboardInterrupt: exit_cmd() 

def help_cmd(message = ''):
    print('''{message}Usage: {name} [parameters]\n
Commands: 
\t {name} : <start system monitoring with default settings>
\t {name} -p | -path : <specify output path> 
\t {name} -c | --collect {time}: <start data collecting>
\t {name} -h | --help : <get this information>
\t {name} -pwd : <get current output path>'''
.format(message = message, name = os.path.basename(sys.argv[0]))) 

if __name__ == '__main__': 
    COMMANDS = dict({'-p': set_path_cmd, '--path': set_path_cmd,
                     '-c': collect_cmd, '--collect': collect_cmd,
                     '-h': help_cmd, '--help': help_cmd,
                     '-pwd': get_path_cmd})  
    set_path_cmd(default = True)
    try:
        COMMANDS[sys.argv[1]]()
    except IndexError: start_cmd()
    except (KeyError, ValueError): help_cmd('Invalid operation: wrong parameter(s)\n')
    except FileNotFoundError: help_cmd('Invalid operation: wrong path\n')

