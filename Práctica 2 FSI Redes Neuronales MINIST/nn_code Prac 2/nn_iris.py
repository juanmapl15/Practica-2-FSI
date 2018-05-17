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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

#70% para entrenamiento
x_data_train=data[0:105, 0:4].astype('f4')
y_data_train=one_hot(data[0:105,4].astype(int),3)

#15% para validacion
x_data_val=data[106:129, 0:4].astype('f4')
y_data_val=one_hot(data[106:129,4].astype(int),3)

#15% para pruebas
x_data_test=data[130:150, 0:4].astype('f4')
y_data_test=one_hot(data[130:150,4].astype(int),3)


#Definir las variables de marcador de posición para las imágenes de entrada
#None significa que el tensor puede contener un número arbitrario de imágenes, siendo cada imagen un vector de longitud 4
x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels


# 4 entradas y con 5 neuronas
#Declaramos la variable w (tensor de pesos)
#Luego declaramos la variable b (tensor de sesgo)

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)


W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

# Definimos nuestro modelo lineal  en la formula del sigmoide Esta función nos devuelve el valor de multiplicar el tensor x * w
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
# Sin embargo, estas estimaciones son un poco difíciles de interpretar, dado que los números que se obtienen pueden ser muy pequeños o muy grandes.
# Softmax: normalizar los valores para que cada fila de la matriz todos sus valores sumen uno.
# así el valor de cada elemento de la matriz esté restringido entre cero y uno
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

# Suma de los cuadrados
loss = tf.reduce_sum(tf.square(y_ - y))

# Tenga en cuenta que la optimización no se realiza en este momento.
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

#100 veces
#batch_xs: lote de imagenes
#batch_ys: etiquestas verdaderas para esas imagenes
for epoch in range(100):
    for jj in range(len(x_data_train) // batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

# argmax: devuleve los indeces de los valores maximos a lo largo de un array
    print ("Validacion")

    # Suma de los cuadrados de nuestros datos que lae pasamos del grafo que esta corriendo
    print ("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
    error1=0
    result = sess.run(y, feed_dict={x: x_data_val})
    j = 1
    for b, r in zip(y_data_val, result):
        print(j, "-->", b, "-->", r)
        if np.argmax(b) != np.argmax(r):
            print("---> Error")
            error1 = error1 + 1
        j += 1
    print ("----------------------------------------------------------------------------------")



print ("Pruebas")

error2=0
result = sess.run(y, feed_dict={x: x_data_test})
i=1
for b, r in zip(y_data_test, result):
    print (i, "-->", b, "-->", r)
    if np.argmax(b) != np.argmax(r):
        print ("---> Error")
        error2=error2+1
    i+=1
print ("Errores totales=", error2)