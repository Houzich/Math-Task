
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import math



def init_codebook(source_vectors,len_codebook):
    #размер вектора (2 элемента в нашем случае)
    len_vector = source_vectors.shape[1]
    #случайно выбираем индексы векторов из исходной выборки для создания кодовой книги
    indexes = np.random.choice(range(source_vectors.shape[0]), len_codebook, replace=False)
    print("indexes:\n",indexes)
    #создаем массив кодовой книги из len_codebook слов по len_vector элемента
    codebook = np.zeros(shape = (len_codebook,len_vector), dtype = source_vectors.dtype)
    
    for i in range(len_codebook):
        codebook[i,:] = source_vectors[indexes[i],:]
    return codebook


SampleArr = np.array([21,10,11,13,15,23,45,67,34,32,12,3,5,23,7,23,6,34,7,34,45])
SampleArr = np.array(range(40))
SampleVectors = np.resize(SampleArr,(SampleArr.size//2,2))
print("SourceVectors:\n",SampleVectors)
CodeBook = init_codebook(SampleVectors,8)
print("CodeBook:\n",CodeBook)



#шаг 1
#возврашает индекс и расстояние до ближайшего из массива векторов (по евклидову расстоянию) 
#к вектору vector
def nearest_ind(samples, vector):
    len_samples = samples.shape[0]
    #для каждого вектора массива samples определяем расстояние до vector
    #для этого создаем матрицy в которой будем хранить расстояния
    distances = np.zeros((len_samples,))
    for i in range(len_samples): 
        distances[i] = np.linalg.norm(samples[i] - vector)
    #определяем наиболее близкий по евклидову растоянию вектор из массива samples
    idx = np.argmin(distances) #номер ближайшего вектора из массива samples
    distance = distances[idx]
    return idx,distance

def lbg_step1(samples, codebook):
    len_samples = samples.shape[0]
    len_codebook = codebook.shape[0]    
    #создаем список Q размерностью исходного массива samples. В каждой ячейке которого
    #запишем соответствующий вектор исходного массива, и ближайший к нему (по евклидову расстоянию) вектор из кодовой книги
    #и само расстояние
    null_vector = np.zeros_like(codebook[0,...])
    QList = [
             {'исходный вектор': null_vector,
              'ближайший вектор': null_vector,
              'расстояние': 0.} 
             for i in range(len_samples)
             ]
    
    for i in range(len_samples):
        QList[i]['исходный вектор'] = samples[i]
        #ищем ближайший вектор из кодовой книги
        idx,distance = nearest_ind(codebook, samples[i])
        QList[i]['ближайший вектор'] = codebook[idx]
        QList[i]['расстояние'] = distance
    #вычисляем ошибку квантования с использованием кодовой книги
    distances = np.array([item['расстояние'] for item in QList])
    mse = np.sum(distances**2)
    return mse

MSE = lbg_step1(SampleVectors, CodeBook)
print("MSE: ",MSE)


#шаг 4

def lbg_step4(source_vectors, codebook):
    len_vectors = source_vectors.shape[0]
    len_codebook = codebook.shape[0]
    
    V = np.zeros_like(codebook)
    S = np.zeros((len_codebook,), dtype=int)
    #создадим пустой список где будем хранить для каждого аппроксимирующего вектора(из кодовой книги) ближайшие вектора из выборки,
    #их сумму, кол-во
    vector = np.zeros_like(codebook[0,...])
    list_test = [
                 {'сумма': np.zeros_like(codebook[0,...]), 'кол-во': 0, 'вектора': [], 'среднее': np.zeros_like(codebook[0,...]),} 
                 for i in range(len_codebook)
                 ]
    #для каждого вектора исходной выборки
    for i in range(len_vectors):        
        #определяем наиболее близкий по евклидову растоянию аппроксимирующий вектор
        distances = np.zeros((len_codebook,)) #матрица расстояний векторов кодовой книги и source_vectors[i]
        for j in range(len_codebook):
            distances[j] = np.linalg.norm(codebook[j] - source_vectors[i])
        idx = np.argmin(distances) #номер ближайшего вектора из кодовой книги
        #складываем ближайшие вектора из выборки, для последующего вычисления нового аппроксимирующего вектора (среднего вектора) 
        V[idx] += source_vectors[i] 
        #считаем кол-во ближайших векторов из выборки, для последующего вычисления нового аппроксимирующего вектора (среднего вектора) 
        S[idx] += 1
        list_test[idx]['сумма'] += source_vectors[i]
        list_test[idx]['кол-во'] += 1
        list_test[idx]['вектора'].append(source_vectors[i])
    #рассчитываем новую кодовую книгу
    new_codebook = np.zeros_like(codebook)
    for i in range(len_codebook): 
        new_codebook[i] = V[i]/S[i]
        list_test[i]['среднее'] = list_test[i]['сумма']/list_test[i]['кол-во']
    return new_codebook

YArr = lbg_step4(SampleVectors, CodeBook)
print("CodeBook:\n",CodeBook)

exit()

<div class="tenor-gif-embed" data-postid="5081425" data-share-method="host" data-width="100%" data-aspect-ratio="1.203883495145631"><a href="https://tenor.com/view/narkomani-smeshno-lol-prikol-narkotiki-gif-5081425">Наркоманы Прикол Радуга GIF</a> from <a href="https://tenor.com/search/narkomani-gifs">Narkomani GIFs</a></div>
<script type="text/javascript" async src="https://tenor.com/embed.js"></script>