import pydicom
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageTk, Image
from skimage import filters
from math import fabs,sqrt
from scipy.signal import convolve2d
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename

#--------------------------Variables globales-------------------------------
filename = ""
intensity = [0]*65536
rows = 0
columns = 0

#----------------------------------Funciones Logica---------------------------------

#--Selecciona el archivo y habilita botones de funciones
def selectFile():
	filename = filedialog.askopenfilename()
	global ds,rows,columns,data
	
	ds = pydicom.dcmread(filename)
	rows = int(ds.Rows)
	columns = int(ds.Columns)


	if filename!="":
		look.config(state="normal")
		histogram.config(state="normal")
		gaussian.config(state="normal")
		raleygh.config(state="normal")
		#fGaussE.config(state="normal")
		mediana.config(state="normal")
		bordes.config(state="normal")
		umbralizacion.config(state="normal")
		kmeans.config(state="normal")

#--Abrir la imagen
def openImage():
	plt.imshow(ds.pixel_array, cmap=plt.cm.gray) #,cmap=plt.cm.gray
	plt.show()

#-----------------------Hacer e imprimir el histograma--------------------------------

def histograma(image):
	img = np.asarray(image)

	tam=len(img)
	tam2=np.amax(img)+1

	histo = [0]*(np.amax(img)+1)

	for i in range(len(image)):
		for j in range(len(image)):
			histo[img[i,j]] = histo[img[i,j]]+1

	histo = np.asarray(histo)

	return histo

def makeHistogram():

	hist = histograma(ds.pixel_array)

	plt.plot(hist)
	plt.show()

#---------------------------------Termina el histograma-------------------------------

#---------------------------------------FILTROS---------------------------------------

#------FUNCION DE CONVOLUCION----
# RECIBE: imagen principal,kernel,escalar

def convolucion(imagen,matriz,escalar):
	imagenM = imagen
	s = len(matriz)#Tamaño del Kernel
	n = int((s-1)/2)#numero de vecinos

	for i in range(rows):
		for j in range(columns):
			if(i<n or i>((rows-1)-n) or j<n or j>((columns-1)-n)):
				imagenM[i][j]=0
			else:
				px=0
				py=0
				aux=0.0
				for kx in range(i-n,i+n+1):
					
					for ky in range(j-n,j+n+1): 

						img = imagen[kx][ky]
						krn = matriz[px][py]

						aux = aux + (img*krn) 

						py += 1

					#end for ky
					px += 1
					py = 0
				#end for kx

				aux = aux/escalar
				aux = int(aux)

				imagenM[i][j] = aux
						
		#end for j
	#end for i
	return imagenM


#----Gaussiano

#--Obtener kernel gaussiano
def getGaussianFilter(neighbours=1, sigma=1):
	N = neighbours*2+1
	X = Y = np.zeros((N, N))

	Y[:] = np.arange(N,dtype=np.float32)-neighbours
	X = np.transpose(Y) 
	
	gaussianFilter = np.zeros((N, N))

	leftSide = (1 / (np.pi * 2 * np.power(sigma,2)))

	aux = -(np.power(X, 2) + np.power(Y, 2)) / (2 * np.power(sigma, 2))
	rightSide = np.exp(aux)

	gaussianFilter = leftSide*rightSide
	factorValue = 1/np.sum(gaussianFilter)

	return gaussianFilter,factorValue

def filtroGaussiano():
	matriz = getGaussianFilter(2)[0]
	escalar = getGaussianFilter(2)[1]

	copia = ds.pixel_array.copy() 
	copia = convolucion(copia,matriz,escalar)

	plt.imshow(copia, cmap=plt.cm.gray)
	plt.show()

#----Termina Gaussiano

#----Rayleigh

#--Obtener kernel rayleigh
def getRayleighFilter(neighbours=1, sigma=1):
	N =  neighbours * 2 + 1
	X = Y = np.zeros((N, N))

	Y[:] = np.arange(N, dtype=np.float32)
	X = np.transpose(Y)

	rayleighFilter = np.zeros((N, N))
	leftSide = np.divide(X*Y, np.power(sigma,4))

	aux = -(np.power(X, 2) + np.power(Y, 2)) / (2 * np.power(sigma, 2))
	rightSide = np.exp(aux)

	rayleighFilter = leftSide*rightSide
	factorValue = 1/np.sum(rayleighFilter)

	return rayleighFilter,factorValue

def filtroRayleigh():
	matriz = getRayleighFilter(2)[0]
	escalar = getRayleighFilter(2)[1]

	copia = ds.pixel_array.copy() 
	copia = convolucion(copia,matriz,escalar)

	plt.imshow(copia, cmap=plt.cm.gray)
	plt.show()

#----Termina Rayleigh

#----Mediana

def ordenar(lista):
	n = len(lista)

	for i in range (n):
		for j in range (0, n-i-1):
			if lista[j] > lista[j+1]:
				aux = lista[j]
				lista[j] = lista[j+1]
				lista[j+1] = aux
	return lista

def filtroMediana(imagen, n = 1):
	for i in range(rows):
		for j in range(columns):
			if i<n or i>((rows-1)-n) or j<n or j>((columns-1)-n):
				imagen[i][j]=0
			else:
				lista =[]*9
				mid = 4

				for x in range(i-n,i+n+1):
					for y in range(j-n,j+n+1):
						lista.append(imagen[x][y])
				lista = ordenar(lista)

				imagen[i][j] = lista[4]

	return imagen

def aplicarMediana():
	copia = ds.pixel_array.copy()

	copia = filtroMediana(copia)

	plt.imshow(copia, cmap=plt.cm.gray)
	plt.show()
#----Termina Mediana

#---------------------------------Terminan los filtros-------------------------------

#-----------------------------------------Sobel--------------------------------------

def aplicarSobel():
	matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Matriz gradiente en X
	matrizSobelX = np.asarray(matrizSobelX)
	matrizSobelY=[[1,2,1],[0,0,0],[-1,-2,-1]] #Matriz gradiente en Y
	matrizSobelY = np.asarray(matrizSobelY)

	copia = ds.pixel_array.copy()

	matriz = getGaussianFilter()[0]
	escalar = getGaussianFilter()[1]

	copia = convolucion(copia,matriz,escalar)
	copia2 = copia.copy()

	gradienteX = convolve2d(copia,matrizSobelX,"same","symm")
	
	gradienteY = convolve2d(copia2,matrizSobelY,"same","symm")

	gradiente = np.abs(gradienteX) + np.abs(gradienteY)

	return gradiente

def hallarBordes():

	resultado = aplicarSobel()

	plt.imshow(resultado, cmap=plt.cm.gray)
	plt.show()

#------------------------------------Terminar Sobel----------------------------------

#----------------------------------------OTSU----------------------------------------

def otsu(hist):
	sum = 0.0
	tam = len(hist)

	for i in range (tam) :
		sum += i*hist[i]

	sumB = 0.0
	wb = 0
	wf = 0

	varmax = 0.0
	umbral = 0
	total = rows*columns
	varBetween=0.0
	mb=0
	mf=0 

	for i in range (tam):
		wb += hist[i]

		if (wb==0):
			continue

		wf = total - wb
		if(wf==0):
			break

		sumB += i*hist[i]

		mb = sumB/wb
		mf = (sum - sumB)/wf

		varBetween = float(wb)*float(wf)*(mb-mf)*(mb-mf)

		if (varBetween > varmax):
			varmax = varBetween
			umbral = i
	#end for
	print(umbral)
	return umbral

def seleccionarBordes(imagen,threshold):
	imagenM = imagen
	tam = len(imagen)

	for i in range(tam):
		for j in range(tam):
			if (imagen[i][j]>threshold):
				imagenM[i][j]=1
			else:
				imagenM[i][j]=0
	return imagen

def aplyOtsu():
	copia = aplicarSobel()
	inte = histograma(copia)

	umbral = otsu(inte)

	deliver = seleccionarBordes(copia,umbral)

	plt.imshow(deliver, cmap=plt.cm.gray)
	plt.show()



#------------------------------------Terminar OTSU-----------------------------------

#---------------------------------------K-MEANS--------------------------------------

def kmedios():
	copia = ds.pixel_array.copy()
	
	plt.imshow(copia)
	plt.show()
#-----------------------------------Terminar k-means---------------------------------










	
def filtroGaussianoEntero():
	matriz = getIntegerValuedGaussianFilter()[0]
	escalar = getIntegerValuedGaussianFilter()[1]

	copia = ds.pixel_array.copy() 
	copia = convolucion(copia,matriz,escalar)

	plt.imshow(copia, cmap=plt.cm.gray)
	plt.show()







#------------------------------Funciones adicionales----------------------------------




#--Obtener kernel gaussiano entero
def getIntegerValuedGaussianFilter(neighbours = 1, sigma = 1):
	N = neighbours*2 + 1

	pascalRow = np.asarray(getKthPascalRow(N - 1), dtype=np.int32)
	pascalRow = np.expand_dims(pascalRow,1)

	extendedPascal = np.resize(pascalRow,(pascalRow.shape[0],pascalRow.shape[0]))

	result = pascalRow * extendedPascal
	scalarFactor = np.sum(result)

	return result,scalarFactor

#--Función para hallar el kernel gaussiano de enteros
def getKthPascalRow(rowNumber):
	if rowNumber == 0:
		return [1, ]

	lastRow = [1, ]
	for R in range(1, rowNumber+1):
		row = []
		row.append(1)
		for C in range(R - 1):
			row.append(lastRow[C] + lastRow[C+1])
		row.append(1)
        
		lastRow = row
	return lastRow












#def clustering(k):





#------------------------------------GUI-----------------------------------
#--Definir ventana
ventanap = Tk()
ventanap.title("DICOM Image Processing")
ventanap.geometry("600x400")

myFrame = Frame(ventanap)

#--Definir botones
msg = StringVar()
inst = Label( myFrame, textvariable=msg)
inst.grid(row=1,column = 1,columnspan=2)
msg.set("Primero seleccione una imagen")

seleccionador = Button(myFrame,text="Seleccionar imagen", width=38, command=selectFile)
seleccionador.grid(row=2,column=1,columnspan=2)

msg1 = StringVar()
basico = Label( myFrame, textvariable=msg1)
basico.grid(row=3,column = 1)
msg1.set("Funciones basicas")

look = Button(myFrame, text = "Mostrar imagen", state = DISABLED, width=18, command=openImage) #
look.grid(row=4,column=1)

histogram = Button(myFrame,text="Hacer histograma", state = DISABLED, width=18, command=makeHistogram)#
histogram.grid(row=4,column=2)

msg2 = StringVar()
basico = Label( myFrame, textvariable=msg2)
basico.grid(row=5,column = 1)
msg2.set("Preprocesamiento")

gaussian = Button(myFrame,text="Filtro Gaussiano", state = DISABLED, width=18, command=filtroGaussiano)#
gaussian.grid(row=6,column=1)

raleygh = Button(myFrame,text="Filtro Raleygh", state = DISABLED, width=18, command=filtroRayleigh)#
raleygh.grid(row=6,column=2)

#fGaussE = Button(myFrame,text="Filtro Gaussiano entero", state = DISABLED, width=18)#, command=filtroGaussianoEntero
#fGaussE.grid(row=6,column=3)

mediana = Button(myFrame,text="Filtro mediana", state = DISABLED, width=18, command=aplicarMediana)#
mediana.grid(row=6,column=3)

msg3 = StringVar()
basico = Label( myFrame, textvariable=msg3)
basico.grid(row=7,column = 1)
msg3.set("Bordes")

bordes = Button(myFrame,text="Bordes(Sobel)", state = DISABLED, width=18, command=hallarBordes)#
bordes.grid(row=8,column=1)

umbralizacion = Button(myFrame,text="Bordes(OTSU)", state = DISABLED, width=18, command=aplyOtsu)#, command=kmedios
umbralizacion.grid(row=8,column=2)

kmeans = Button(myFrame,text="K-means", state = DISABLED, width=18)#, command=kmedios
kmeans.grid(row=8,column=3)

myFrame.pack()

ventanap.mainloop()
