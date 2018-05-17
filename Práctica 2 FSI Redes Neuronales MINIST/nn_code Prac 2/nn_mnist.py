import gzip
import pickle as cPickle
import sys


import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
# Devuelve una matriz de la quinta columna pasada a binario para tener la codificacion de los tipos
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

f=  gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='iso-8859-1')
f.close()

# Entreno
train_x, train_y = train_set
train_y=one_hot(train_y.astype(int), 10)

#Validación
val_x, val_y = valid_set
val_y=one_hot(val_y.astype(int), 10)

# Pruebas
test_x, test_y = test_set
test_y=one_hot(test_y.astype(int), 10)


#Definir las variables de marcador de posición para las imágenes de entrada
#None significa que el tensor puede contener un número arbitrario de imágenes, siendo cada imagen un vector de longitud 4
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels


#Declaramos la variable w (tensor de pesos)
#Luego declaramos la variable b (tensor de sesgo)

# Esto hace referencia a la entrada
W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# Esto hace referencia a la salida de las neuronas
W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

# Definimos nuestro modelo lineal  en la formula del sigmoide Esta función nos devuelve el valor de multiplicar el tensor x * w
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)

# Sin embargo, estas estimaciones son un poco difíciles de interpretar, dado que los números que se obtienen pueden ser muy pequeños o muy grandes.
# Softmax: normalizar los valores para que cada fila de la matriz todos sus valores sumen uno.
# así el valor de cada elemento de la matriz esté restringido entre cero y uno
#Normalizamos los valores de la matriz [0,0,0,0,1,0,0,0]
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

#Suma de los cuadrados
loss = tf.reduce_sum(tf.square(y_ - y))

# Tener en cuenta que la optimización no se realiza en este momento.
# Simplemente agregamos el objeto optimizador al gráfico TensorFlow para su posterior ejecución.
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

#Inicializar las variables para pesos y sesgos deben inicializarse antes de comenzar a optimizarlas.
init = tf.global_variables_initializer()

#Una vez espcificados todos los elementos de nuestro modelo, podemos ya crear el grafo.
# Para ello tenemos que crear una sesión para ejecutar luego el grafo.

sess = tf.Session()
sess.run(init)


print ("----------------------")
print ("   Start training...  ")
print ("----------------------")

batch_size = 20

entrenografic=[]
validaciongrafic=[]
errorsvalidagrafic=0
erroranterior=0.
erroractual=100.
epoch=0

#Cuando la diferencia con respecto al error anterior
while (abs(erroractual-erroranterior)) > 0.01:
    for jj in range(len(train_x) // batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Error del entreno
    erroractual = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    entrenografic.append(erroractual)
    if epoch >1: # te aseguras que hay minimo 2 errores para coger el anterior al actual
        erroranterior=entrenografic[-2]

    print("Validacion")
    print("Epoch #:", epoch, "Error: ", erroractual)# Imprime el error del entreno

    errorsvalidagrafic = sess.run(loss, feed_dict={x: val_x, y_: val_y})
    validaciongrafic.append(errorsvalidagrafic)

    print("Epoch #:", epoch, "Error: ", errorsvalidagrafic)
    epoch+=1


    # result = sess.run(y, feed_dict={x: val_x})
    # for b, r in zip(val_y, result):
    #    print b, "-->", r
    # print "----------------------------------------------------------------------------------"



print ("Pruebas")
error=0
acierto=0
result = sess.run(y, feed_dict={x: test_x})

for b, r in zip(test_y, result):
    #print( b, "-->", r)
    if np.argmax(b) != np.argmax(r):
        #print("---> Error")
        error +=1
    else:
        acierto +=1
#print ("Errores totales=", error)
#print ("Aciertos totales=", acierto)

print ("Porcentaje test errores=", (error*100)/(error+acierto),"%")
print ("Porcentaje test aciertos=",(acierto*100)/(error+acierto),"%")







# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.title("Entreno")
plt.plot(entrenografic)# Mostramos la gráica de los errores del entreno
#plt.plot(validaciongrafic) # Mostramos la gráica de los errores de la validacion
plt.show()

plt.title("Validacion")
plt.plot(validaciongrafic)# Mostramos la gráica de los errores del entreno
plt.plot(validaciongrafic) # Mostramos la gráica de los errores de la validacion
plt.show()

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print (test_y[2])


# TODO: the neural net!!


