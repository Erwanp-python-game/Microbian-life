import pygame
from math import *
import numpy as np
from pygame.locals import *
from random import *
import matplotlib.pyplot as plt
from copy import*
for i in range(1,10):
	a=0
	for j in range(0,i):
		a=a+abs(cos(2*pi*(j+0.5)/i))
	b=0
	for j in range(0,i):
		b=b+abs(cos(2*pi*(j)/i))
	print(max(a,b),i)



L=600
pygame.init()
fenetre = pygame.display.set_mode((L+200, L))
print(pygame.time.get_ticks())
courantX=np.full((L,L),0.0)
courantY=np.full((L,L),0.0)

courantX0=np.full((3*L,3*L),0.0)
courantY0=np.full((3*L,3*L),0.0)

x,y = np.meshgrid(np.linspace(-30,30,3*L),np.linspace(-30,30,3*L))
speed=3
for i in range(0,5):
	xR=randint(-10,10)
	yR=randint(-10,10)
	I=(-1)**randint(0,1)
	Rayon=randint(4,15)
	
	z2=speed*(I*(x-xR)/(((x-xR)**2+(y-yR)**2)**0.5))*np.exp(-((((x-xR)**2+(y-yR)**2)-Rayon**2)**2)/(60**2))
	z1=speed*(-I*(y-yR)/(((x-xR)**2+(y-yR)**2)**0.5))*np.exp(-((((x-xR)**2+(y-yR)**2)-Rayon**2)**2)/(60**2))
	courantX0=np.add(courantX0,np.array(z1))
	courantY0=np.add(courantY0,np.array(z2))

for i in range(0,3):
	for j in range(0,3):
		courantX=np.add(courantX0[i*L:(i+1)*L,j*L:(j+1)*L],courantX)
		courantY=np.add(courantY0[i*L:(i+1)*L,j*L:(j+1)*L],courantY)
print(pygame.time.get_ticks())
# courantX=courantX0[L:2*L,L:2*L]
# courantY=courantY0[L:2*L,L:2*L]

subL=60
dt=0.1
diffu=0.1
nourriture=np.random.uniform(size=subL**2)*4
flux=np.full((subL**2,subL**2),0.0)

for i in range(0,subL**2):
		flux[i][i]=(1/dt)-abs(courantY[i%subL][i//subL])-abs(courantX[i%subL][i//subL])-4*diffu/dt
		
		
		flux[i][(i+1)%(subL**2)]=0.5*abs(courantX[10*((i)//subL)][10*(((i+1)%(subL**2))%subL)]-abs(courantX[10*((i)//subL)][10*(((i+1)%(subL**2))%subL)]))+diffu/dt
		flux[i][(i-1)%(subL**2)]=0.5*abs(courantX[10*((i)//subL)][10*(((i-1)%(subL**2))%subL)]+abs(courantX[10*((i)//subL)][10*(((i-1)%(subL**2))%subL)]))+diffu/dt

		
		flux[i][(i+subL)%(subL**2)]=0.5*abs(courantY[10*(((i+subL)%(subL**2))//subL)][10*(i%subL)]-abs(courantY[10*(((i+subL)%(subL**2))//subL)][10*((i)%subL)]))+diffu/dt
		flux[i][(i-subL)%(subL**2)]=0.5*abs(courantY[10*(((i-subL)%(subL**2))//subL)][10*(i%subL)]+abs(courantY[10*(((i-subL)%(subL**2))//subL)][10*((i)%subL)]))+diffu/dt



flux=flux*dt

q=0
trace=0
clock = pygame.time.Clock()
CO2=30000
O2=30000
C_dissou=30000

images_cell={}
images_cell['M']=pygame.image.load('M.png')
images_cell['P']=pygame.image.load('P.png')
images_cell['G']=pygame.image.load('G.png')
images_cell['N']=pygame.image.load('N.png')
images_cell['Y']=pygame.image.load('Y.png')
images_cell_center=['M','P','G']
images_cell_end=['N','Y']#,'Y','P','C']

all_codes=[]
nbc={}
nbc['M']=randint(0,200)
nbc['P']=randint(0,200)
nbc['G']=randint(0,200)
nbc['N']=randint(0,200)
nbc['Y']=randint(0,200)

fleche=pygame.image.load('fleche.png')
fond=pygame.Surface((L,L),pygame.SRCALPHA, 32)
fond.fill((0,50,200))
for i in range(0,30):
	for j in range(0,30):
		if courantX[i*20][j*20]>0:
			angle=np.arctan(courantY[i*20][j*20]/courantX[i*20][j*20])*360/(2*pi)+180
		else:
			angle=np.arctan(courantY[i*20][j*20]/courantX[i*20][j*20])*360/(2*pi)
		flecheR=pygame.transform.rotate(fleche,angle)
		S=int((10/speed)*(courantX[i*20][j*20]**2+courantY[i*20][j*20]**2)**0.5)
		flecheR=pygame.transform.scale(flecheR,(S,S))
		fond.blit(flecheR,(20*i,20*j))
fond0=fond.copy()
		
def AngleReturn(x,y):
	if x<0:
		return np.arctan(y/x)*(180/pi)+180
	else:
		return np.arctan(y/x)*(180/pi)

def meanAngle(a1,a2):
	a1=a1/(180/pi)
	a2=a2/(180/pi)
	return AngleReturn(0.9*cos(a1)+0.1*cos(a2),0.9*sin(a1)+0.1*sin(a2))

def show_food(actif):
	if actif==1:
		if trace==1:
			fond2=fond.copy()
		else:
			fond2=fond0.copy()
		listim=[]
		im=pygame.Surface((L//subL,L//subL),pygame.SRCALPHA, 32)
		for i in range(0,subL**2):
			im.fill((0,min(max(int(200*nourriture[i]),0),255),200,100))
			
			listim.append((im.copy(),((L//subL)*(i//subL),(L//subL)*(i%subL))))
		
		fond2.blits(listim)
		return fond2
	else:
		return fond

def grad(I,yeux):
	X=np.full((2*yeux+1,2*yeux+1),0.0)
	for i in range(-yeux,yeux+1):
		for j in range(-yeux,yeux+1):
			X[i+yeux][j+yeux]=nourriture[(I+i+60*j)%3600]-nourriture[(I)%3600]*(1+uniform(-5,5)/yeux**2)# décroit avec distance mais compensé yeux
	pos=np.unravel_index(np.argmax(X, axis=None), X.shape)
	return (pos[0]-yeux,pos[1]-yeux)
	
All_Org=[]
class Organism():
	
	def __init__(self,x,y,code,Img):
		self.xc=x
		self.yc=y
		self.vx=0
		self.vy=0
		self.age=0
		self.code=code
		self.im=Img
		self.type_nourr=0
		self.type_photo=0
		self.gras=0
		self.angle=0
		self.yeux=0
		self.nageoire=0
		seedn=0
		self.fast=0
		addL=0
		for i in code:
			for j in i:
				if 'M'==j:
					self.type_nourr+=1
					self.fast=0.1
				if 'P'==j:
					self.type_photo+=1
				if 'G'==j:
					self.gras+=0.5
				if 'Y'==j:
					self.yeux+=1
				if 'N'==j:
					self.nageoire+=1
			addL=addL+2*(len(i)-1)
		addL=addL-2*self.nageoire-2*self.yeux
		for j in range(1,len(code)):
			seedn+=nbc[self.code[j][0]]
		seed(seedn)
		self.color=(randint(0,255),randint(0,255),randint(0,255))
		seed()
		self.sym=self.code[0][0]
		self.size=(len(code)-2)*self.sym+1+addL
		self.type_nourr=max((self.type_nourr-1)*self.sym+1,0)# 
		self.type_photo=max((self.type_photo-1)*self.sym+1,0)# gras pas assez efficace
		self.gras=max((self.gras-0.5)*self.sym+0.5,0)
		self.yeux=int(max((self.yeux-1)*self.sym+1,0)*np.heaviside(self.fast,0))
		self.nageoire=int(max((self.nageoire-1)*self.sym+1,0)*np.heaviside(self.fast,0))
		self.stockedCO2=250+50*self.size#300 de base
		

		
	def buil_Im(self):
		Lm=max(30,20*(len(self.code)),20*max(len(elem) for elem in self.code))
		self.im=pygame.Surface((Lm,Lm),pygame.SRCALPHA, 32)# take width into account
		
		for j in range(0,self.sym):
			for i in range(1,len(self.code)):
				A=j*2*pi/self.sym
				X=int((i-1)*10*sin(A))+Lm//2-5
				Y=int((i-1)*10*cos(A))+Lm//2-5
				self.im.blit(images_cell[self.code[i][0]],(X,Y))
				for k in range(1,len(self.code[i])):
					deltaX=int((k)*10*cos(A))
					deltaY=-int((k)*10*sin(A))
					if self.code[i][k] not in images_cell_end:
						self.im.blit(images_cell[self.code[i][k]],(X+deltaX,Y+deltaY))
						self.im.blit(images_cell[self.code[i][k]],(X-deltaX,Y-deltaY))
					else:
						Im=pygame.transform.rotate(images_cell[self.code[i][k]],A*180/pi)
						shift=(Im.get_width()//2)-5
						Ir=pygame.transform.flip(images_cell[self.code[i][k]],True,True)
						self.im.blit(Im,(X+deltaX-shift,Y+deltaY-shift))
						self.im.blit(pygame.transform.rotate(Ir,A*180/pi),(X-deltaX-shift,Y-deltaY-shift))
		self.im.fill((255,255,255,215),special_flags=BLEND_RGBA_MULT)

		if len(all_codes)==len(all_im)+1 and self.code==all_codes[-1]:
			all_im.append(self.im)
		
	def move_eat(self):
		global CO2,O2,nourriture,All_Org
		I1=int(self.yc)//10+60*(int(self.xc)//10)
		G=grad(I1,self.yeux+1)
		G=G/((G[0]**2+G[1]**2)**0.5+0.0001)
		self.vx=(0.5*self.vx+0.5*courantY[int(self.xc)][int(self.yc)]/(log(self.size/10+3)*(3*self.nageoire+1)))+(8+2*self.nageoire+2*self.yeux)*dt*G[1]*np.heaviside(self.fast,0)/log(self.size/20+3)+(1-np.heaviside(self.fast,0))*np.random.normal(0,0.2)# div norme de G
		self.vy=(0.5*self.vy+0.5*courantX[int(self.xc)][int(self.yc)]/(log(self.size/10+3)*(3*self.nageoire+1)))+(8+2*self.nageoire+2*self.yeux)*dt*G[0]*np.heaviside(self.fast,0)/log(self.size/20+3)+(1-np.heaviside(self.fast,0))*np.random.normal(0,0.2)# viv v par taille
		self.xc=(self.xc+self.vx)%L
		self.yc=(self.yc+self.vy)%L
		consumed=0
		if self.type_nourr>0:
			consumed=max(min(nourriture[I1],10*self.type_nourr,10*self.type_nourr*O2/15000),0)
			nourriture[I1]=nourriture[I1]-consumed
			CO2=CO2+consumed/2
			O2=O2-consumed
		if self.type_photo>0:
			consumed=max(min(10*self.type_photo,10*self.type_photo*CO2/15000),0)
			nourriture[I1]=nourriture[I1]+consumed/2
			CO2=CO2-consumed
			O2=O2+consumed/2
		
		self.stockedCO2=self.stockedCO2+consumed/2
		self.age=self.age+1+0.5*(9+0.05*self.size-consumed*(self.sym+0.2)/(self.sym))/(1+self.gras)#à voir
		self.I1=I1
		self.angle=meanAngle(self.angle,AngleReturn(self.vx,self.vy))
		
		if self.stockedCO2>500+100*self.size and randint(0,10)==0:
			if randint(0,7)==0:
				#print(self.code)
				CM=mutation(deepcopy(self.code))
				#print(CM,self.code)
				All_Org.append(Organism((self.xc+np.random.normal(0,5))%600,(self.yc+np.random.normal(0,5))%600,CM,self.im))
				All_Org[-1].buil_Im()
				nb[all_codes.index(CM)]+=1
			else:
				All_Org.append(Organism((self.xc+np.random.normal(0,5))%600,(self.yc+np.random.normal(0,5))%600,self.code,self.im))
				nb[all_codes.index(self.code)]+=1
			self.stockedCO2=self.stockedCO2-250-50*self.size
			
			
				
	
	def draw(self,trace):
		global fenetre,all_codes
		if trace==1:
			pygame.draw.circle(fond,self.color,(int(self.xc-1),int(self.yc-1)),2)
		#pygame.draw.circle(fenetre,(self.type_nourr*255,(1-self.type_nourr)*255,0),(int(self.xc),int(self.yc)),4)
		fenetre.blit(pygame.transform.rotate(self.im,-self.angle-90),(int(self.xc-self.im.get_width()//2),int(self.yc-self.im.get_height()//2)))
			
	
	def alive(self):
		return (self.age<2800+np.random.normal(0,100)+200*self.size)
		
	def release(self):
		global CO2,O2,nourriture
		if randint(0,1)==1:
			nourriture[self.I1]=nourriture[self.I1]+self.stockedCO2
			O2=O2+self.stockedCO2
		else:
			CO2=CO2+self.stockedCO2
		nb[all_codes.index(self.code)]-=1
		
		

all_codes.append([[1],['M']])
all_codes.append([[1],['P']])
all_im=[pygame.Surface((30,10),pygame.SRCALPHA, 32),pygame.Surface((30,10),pygame.SRCALPHA, 32)]
col=[]

def update_colors():
	global col
	col=[]
	for code1 in all_codes:
		seedn=0
		for j in range(1,len(code1)):
			seedn+=nbc[code1[j][0]]
		seed(seedn)
		col.append((randint(0,255)/255,randint(0,255)/255,randint(0,255)/255))
update_colors()
nb=[0,0]

def showcode(code):
	a=''
	for i in code:
		a=a+'/'
		for j in i:
			a=a+str(j)
	return a

def mutation(code_g):# assurer qu'une mutation ait lieu
	muted=0
	code_r=deepcopy(code_g)
	while muted==0:
	
		R=randint(-2,10)
		
		if R>8:
			Muta=randint(1,len(code_g)-1)
			code_r[Muta][0]=choice(images_cell_center)
			
		if R<=8 and R>6:
			if randint(0,1)>=1:
				a=1
				for i in range(2,len(code_r)):
					a=min((i-1)*abs(tan(pi/(code_r[0][0]+1)))-len(code_r[i])+0.5,a)# là y a un pb
				print('sym',a,code_r[0][0],code_g)
				if len(code_r)>2 and (code_r[0][0]==1 or a>=0) and len(code_r[1])==1:
					code_r[0][0]+=1
					muted=1
					print('sym_app')
				
			else:
				code_r[0][0]=max(code_r[0][0]-1,1)
				muted=1
			
		if R<=6 and R>1:
	
			if len(code_r)==2 and len(code_r[1])==1:
				code_r[0][0]=choices([1,2,3,4,5,6],[6,5,4,3,2,1],k=1)[0] # faire rétrécir
			code_r.append([choice(images_cell_center)])
			muted=1
		
		if R<=1:# check angle
				print("largeur")
				Muta=randint(1,len(code_g)-1)
				if (Muta-1)*abs(tan(pi/(code_g[0][0])))-1>len(code_g[Muta]) or code_g[0][0]<3:
					if randint(0,len(images_cell_center)+len(images_cell_end))>len(images_cell_end) or (code_r[Muta][-1] in images_cell_end):
						code_r[Muta].insert(0,choice(images_cell_center))
					else:
						code_r[Muta].append(choice(images_cell_end))
					muted=1
		
		if (code_r in all_codes)==False:
			all_codes.append(code_r)
			update_colors()
			nb.append(0)
			print(showcode(code_r))
			return code_r
			
		else:
			return code_r

font = pygame.font.Font('freesansbold.ttf', 13)
def show_species():
	back=pygame.Surface((200,L), 32)
	back.fill((0,0,0))
	U=20-curseur*50
	fenetre.blit(back,(L,0))
	for i,j in enumerate(reversed(all_codes)):
		U=U+50
		T=showcode(j)
		text = font.render(T, True, (255,255,255))
		textRect = text.get_rect()
		textRect.topleft = (L+20, U)
		fenetre.blit(text,textRect)
		IM=pygame.transform.rotate(all_im[len(all_im)-i-1],90)
		IM=IM.convert(back)
		Ll=int(50*IM.get_width()/IM.get_height())
		fenetre.blit(pygame.transform.scale(IM,(Ll,50)),(textRect[0]+textRect.width+20,U))
		



for i in range(0,10):
	R=randint(0,1)
	if R==1:
		code=[[1],['M']]
	else:
		code=[[1],['P']]
	

	All_Org.append(Organism(randint(0,L-1),randint(0,L-1),code,0))
	All_Org[-1].buil_Im()
	nb[all_codes.index(code)]+=1
	
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)	
ax2 = fig.add_subplot(2, 1, 2)
graphe=0
I=-1
fenetre.blit(fond,(0,0))
curseur=0
while q==0:
	T1=pygame.time.get_ticks()
	I+=1

	#pygame.time.wait(100)

	nourriture=np.dot(flux,nourriture)
	nourriture=np.absolute(nourriture)*(O2/np.absolute(nourriture).sum())
	
	for event in pygame.event.get():
		if event.type == QUIT:
			q=1
		KEY=pygame.key.get_pressed()
		if KEY[K_SPACE]:
			trace=(trace+1)%2
			print(trace)
		if event.type==MOUSEBUTTONUP:
			if event.button==5:
				curseur=(curseur+1)
			if event.button==4:
				curseur=(curseur-1)

			

	Im=show_food(1)
	fenetre.blit(Im,(0,0))
	
	for i in All_Org:
		i.move_eat()
		i.draw(trace)
		if i.alive()==False:
			i.release()
			All_Org.remove(i)
	
	if (I%100)==0 and graphe==1:
		ax2.scatter(I,O2,c='blue')
		ax2.scatter(I,CO2,c='red')
		for i in range(0,len(col)):
			ax1.scatter(I,nb[i],color=col[i])
			
		
		plt.pause(0.001)
	if (I%100)==1:
		print(pygame.time.get_ticks()-T1,O2+2*CO2+nourriture.sum()-300*2*10,np.sum(nb))
	show_species()
	pygame.display.flip()
	clock.tick(30)

plt.show()
