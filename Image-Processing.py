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
from tkinter import ttk
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

	
	msg11.set(str(ds.PatientID))
	msg13.set(str(ds.Rows))
	msg15.set(str(ds.Columns))
	msg17.set(str(ds.PixelSpacing))

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
	
	k = tamKernel.get()
	tamano=int(k[:k.index("x")])
	tamano = int((tamano-1)/2)

	matriz = getGaussianFilter(tamano)[0]
	escalar = getGaussianFilter(tamano)[1]

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

	k = tamKernel.get()
	tamano=int(k[:k.index("x")])
	tamano = int((tamano-1)/2)

	matriz = getRayleighFilter(tamano)[0]
	escalar = getRayleighFilter(tamano)[1]

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
	
	vecinos = n
	tamKrn = (2*n)+1
	tamArr = tamKrn**2
	

	for i in range(rows):
		for j in range(columns):
			if i<n or i>((rows-1)-n) or j<n or j>((columns-1)-n):
				imagen[i][j]=0
			else:
				lista =[]*tamArr

				for x in range(i-n,i+n+1):
					for y in range(j-n,j+n+1):
						lista.append(imagen[x][y])
				lista = ordenar(lista)

				mid = int((len(lista)-1)/2)

				imagen[i][j] = lista[mid]

	return imagen

def aplicarMediana():
	copia = ds.pixel_array.copy()

	k = tamKernel.get()
	tamano=int(k[:k.index("x")])
	tamano = int((tamano-1)/2)

	copia = filtroMediana(copia,tamano)

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

def definirCentroides(k):
    contador=0

    centroides= ds.pixel_array.copy()

    cn = []

    for i in range(k):
        cn.append(int(255/k)*i)

    
    while(contador<2):
        arrayCn = []
        
        for n in range(k):
            arrayCn.append([])
        for i in range(rows):
            for j in range(columns):
                distancias=[]
                for n in range(k):
                    distancias.append(fabs(cn[n]-ds.pixel_array[i][j]))

                index=distancias.index(np.amin(distancias))
                arrayCn[index].append(ds.pixel_array[i][j])

                centroides[i][j]=index*int(255/k)
            #print("row: " + str(i)+" columns: "+str(j))

        iguales=True                
        for n in range(k):
            if(cn[n]!=int(np.mean(arrayCn[n]))):
                cn[n]=int(np.mean(arrayCn[n]))
                iguales=False
        
        if(iguales):
            contador+=1

    return centroides

def kmedios():
	copia = ds.pixel_array.copy()

	centroides = numCents.get()
	centroides = int(centroides)

	copia = definirCentroides(centroides)
	
	plt.imshow(copia)
	plt.show()
#-----------------------------------Terminar k-means---------------------------------


#------------------------------------GUI-----------------------------------
#--Definir ventana
ventanap = Tk()
ventanap.title("DICOM Image Processing")
ventanap.geometry("700x500")

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

look = Button(myFrame, text = "Mostrar imagen", state = DISABLED, width=18, command=openImage)
look.grid(row=4,column=1)

histogram = Button(myFrame,text="Hacer histograma", state = DISABLED, width=18, command=makeHistogram)
histogram.grid(row=4,column=2)

msg2 = StringVar()
proce = Label( myFrame, textvariable=msg2)
proce.grid(row=5,column = 1)
msg2.set("Preprocesamiento")

msg3 = StringVar()
krntam = Label( myFrame, textvariable=msg3)
krntam.grid(row=6,column = 1)
msg3.set("Tamaño del kernel: ")

tamKernel = ttk.Combobox(myFrame, values=("3x3", "5x5", "7x7", "9x9"),state="readonly")
tamKernel.set("3x3")
tamKernel.grid(row=6,column=2)

msg4 = StringVar()
fil = Label( myFrame, textvariable=msg4)
fil.grid(row=7,column = 1)
msg4.set("Filtros")

gaussian = Button(myFrame,text="Filtro Gaussiano", state = DISABLED, width=18, command=filtroGaussiano)
gaussian.grid(row=8,column=1)

raleygh = Button(myFrame,text="Filtro Raleygh", state = DISABLED, width=18, command=filtroRayleigh)
raleygh.grid(row=8,column=2)

mediana = Button(myFrame,text="Filtro mediana", state = DISABLED, width=18, command=aplicarMediana)
mediana.grid(row=8,column=3)

msg5 = StringVar()
bor = Label( myFrame, textvariable=msg5)
bor.grid(row=9,column = 1)
msg5.set("Bordes")

bordes = Button(myFrame,text="Bordes(Sobel)", state = DISABLED, width=18, command=hallarBordes)
bordes.grid(row=10,column=1)

umbralizacion = Button(myFrame,text="Bordes(OTSU)", state = DISABLED, width=18, command=aplyOtsu)
umbralizacion.grid(row=10,column=2)

msg6 = StringVar()
bor = Label( myFrame, textvariable=msg6)
bor.grid(row=11,column = 1)
msg6.set("Segmentación")

msg7 = StringVar()
bor = Label( myFrame, textvariable=msg7)
bor.grid(row=12,column = 1)
msg7.set("Número de centroides: ")

numCents = ttk.Combobox(myFrame, values=("2", "3", "4", "5"),state="readonly")
numCents.set("2")
numCents.grid(row=12,column=2)

kmeans = Button(myFrame,text="K-means", state = DISABLED, width=18, command=kmedios)
kmeans.grid(row=12,column=3)

msg8 = StringVar()
sep = Label( myFrame, textvariable=msg8)
sep.grid(row=13,column = 1, columnspan=3)
msg8.set("------------------------------------------------------------------------------------------")

msg9 = StringVar()
bor = Label( myFrame, textvariable=msg9)
bor.grid(row=14,column = 2)
msg9.set("INFORMACIÓN")

msg10 = StringVar()
idp = Label( myFrame, textvariable=msg10)
idp.grid(row=15,column = 1)
msg10.set("ID Paciente: ")

msg11 = StringVar()
idp2 = Label( myFrame, textvariable=msg11)
idp2.grid(row=15,column = 3)

msg12 = StringVar()
fila = Label( myFrame, textvariable=msg12)
fila.grid(row=16,column = 1)
msg12.set("Filas: ")

msg13 = StringVar()
fila2 = Label( myFrame, textvariable=msg13)
fila2.grid(row=16,column = 3)

msg14 = StringVar()
columna = Label( myFrame, textvariable=msg14)
columna.grid(row=17,column = 1)
msg14.set("Columnas: ")

msg15 = StringVar()
columna2 = Label( myFrame, textvariable=msg15)
columna2.grid(row=17,column = 3)

msg16 = StringVar()
ps = Label( myFrame, textvariable=msg16)
ps.grid(row=18,column = 1)
msg16.set("Pixel spacing: ")

msg17 = StringVar()
ps2 = Label( myFrame, textvariable=msg17)
ps2.grid(row=18,column = 3)

myFrame.pack()

ventanap.mainloop()
