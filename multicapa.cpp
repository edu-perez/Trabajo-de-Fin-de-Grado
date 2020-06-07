#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include "aleatorios.h"
#include <time.h>


using namespace std;
#define pi  3.14159265359


#define m   35 //nº neuronas en las capas ocultas. Recuerda que hay (ncapas-2) capas ocultas.
#define npat    60 //nº patrones (ejemplos) para el train
#define n_test  300//nº ejemplos para el test. Pongo muchos para pintar bien toda la función como salida de red luego.
#define n   1   //nº neuronas capa de entrada (nº de variables de la función vaya. Normalmente 1 (sin(x), x, ...) aunq con funciones lógicas como XOR se requieren 2 entradas
#define ncapas  6   //nº capas del perceptrón. Recuerda q 1 es de entrada, otra es de salida y el resto ocultas.
//#define b   10.0    //b es el parámetro en la función de activ. tal que exp(-b*x)

double sigmoide(double x, double b);
double derivada_sigm(double x, double b);
void paso_red(double w[m][m][ncapas], double u[m][ncapas], int capa[ncapas], double y[m][ncapas][npat], int p);
void desordenar(double x[n][npat]);
void vector_x(double a,double b, double x[n][npat]);
void paso_red_test(double w[m][m][ncapas], double u[m][ncapas], int capa[ncapas], double y1[m][ncapas][n_test], int p);


int main (void)
{
    
    ofstream err ("error_overfitting_60_4.txt");//fichero donde se observa la evolución del error total
    ofstream grafica_ent("ent_60.txt");//fichero donde dibujo w l los puntos de entrenamiento.
    ofstream grafica_final("overf60_4.txt");//fichero con los ptos para plotear la función q la red produce.
    ofstream e_ruido("e_ruido_overf_60_4.txt");
    
    int verdad,veces,k,i,j,p,contador;
    int capa[ncapas];
    double w[m][m][ncapas], u[m][ncapas], y[m][ncapas][npat],delta[m][ncapas];
    
    double lim_inferior, lim_superior;
    lim_inferior=-1;
    lim_superior=1;
    
    double x[n][npat];
    
    vector_x(lim_inferior,lim_superior,x);//aquí se forma el vector con los inputs de la red, desordenados.
    
    
    
    for(i=0;i<npat;i++)//Aaquí está el vector entrada de la red
        y[0][0][i]=x[0][i];
    
    for (i=1;i<(ncapas-1);i++)//capas ocultas tienen m neuronas
        capa[i]=m;
    
    capa[0]=n;//capa de entrada tiene n neuronas, q generalmente será 1
    capa[ncapas-1]=1;//capa de salida siempre tiene 1 neurona.
        
    double   result[npat];//entrenar AND: {1,-1,-1,-1}    entrenar OR: {1,1,1,-1}    entrenar XOR:{-1,1,1,-1}
    k=32185;
    dranini_(&k);
    
    for(i=0;i<npat;i++)
    {
        result[i]=sin(pi*y[0][0][i])+(dranu_()*0.2-0.1);  //esto es el ruido.
    }
    
    double y0,dery0[m],z0,uy[m],nu,derz0,sum;
    double z[npat];//salida de la red (de la neurona de salida vaya)
    
    double b;//b es el parámetro en la función de activ. tal que exp(-b*x)
    b=10.0;
    
    double error[npat],errort, precision,error_ruido[npat], errort_ruido;
    precision=0.001;

    double inc_ant_u[m][ncapas], inc_ant_w[m][m][ncapas],tasa,incremento;//esto es para el término denominado momento, q acelera la convergencia.
    tasa=0.8;
    
    for(k=0;k<ncapas;k++)//inicializo todos los elementos de los tensores, aunq alguno de ellos no se vayan a utilizar. ¿pq?
    {
        for(j=0;j<m;j++)
        {
            for(i=0;i<m;i++)
            {
                inc_ant_w[j][i][k]=0.0;
            }
            
            inc_ant_u[j][k]=0.0;
        }
    }

    //ofstream prueba("prueba.txt");
    
    //INICIALIZACIÓN DE PESOS Y UMBRALES.
    k=32185;
    dranini_(&k);
    
    for(k=0;k<ncapas;k++)//inicializo todos los elementos de los tensores, aunq alguno de ellos no se vayan a utilizar.
    {
        for(j=0;j<m;j++)
        {
            for(i=0;i<m;i++)
            {
                w[j][i][k]=dranu_()*0.01-0.005;//pesos son aleatorios € (0,0.5)
            }
            
            u[j][k]=dranu_()*0.01-0.005;
        }
    }

    nu=0.001;//tasa de aprendizaje
    
    cout<<endl;
    verdad=0;
    veces=0;
    
    //ENTRENAMIENTO DE LA RED
    while((verdad==0)&&(veces<1560000))
    {
        veces++;//esto es simplemente un contador, por si queremos ver cuanto tarda en entrenarse
        
        p=floor((dranu_()*npat));
        
        paso_red(w,u,capa,y,p);
        
        z[p]=y[0][ncapas-1][p];

            error[p]=abs(result[p]-z[p]);
            
            if(error[p]>precision)
            {
                derz0=1.0;//la func. act. de la capa (neurona) de salida es la función identidad.
                delta[0][ncapas-2]=result[p]-z[p];
                incremento=nu*delta[0][ncapas-2]*derz0;
                u[0][ncapas-2]=u[0][ncapas-2]+incremento+tasa*inc_ant_u[0][ncapas-2];
                inc_ant_u[0][ncapas-2]=incremento;
                
                for(k=(ncapas-1);k>1;k=k-1)
                {
                    //cout<<k<<endl;
                    for(i=0;i<capa[k-1];i++)
                    {
                        sum=0.0;
                        for(j=0;j<capa[k];j++)
                        {
                            incremento=nu*y[i][k-1][p]*delta[j][k-1];
                            w[j][i][k-1]=w[j][i][k-1]+incremento+tasa*inc_ant_w[j][i][k-1];
                            inc_ant_w[j][i][k-1]=incremento;
                            sum=sum+delta[j][k-1]*w[j][i][k-1];
                        }
                        delta[i][k-2]=derivada_sigm(y[i][k-1][p],b)*sum;
                        incremento=nu*delta[i][k-2];
                        u[i][k-2]=u[i][k-2]+incremento+tasa*inc_ant_u[i][k-2];
                        inc_ant_u[i][k-2]=incremento;
                    }
                }
                
                k=1;//el último hay hacerlo aparte ya que solo hay que calcular el increm. de los primeros pesos. Lo demás ya está
                for(i=0;i<capa[k-1];i++)//este for en realidad es como si no existise vaya, ya q la capa[0] generalmente tendrá 1 neurona
                {
                    //sum=0.0;
                    for(j=0;j<capa[k];j++)
                    {
                        incremento=nu*y[i][k-1][p]*delta[j][k-1];
                        w[j][i][k-1]=w[j][i][k-1]+incremento+tasa*inc_ant_w[j][i][k-1];
                        inc_ant_w[j][i][k-1]=incremento;
                    }
                }
            }
                
                
        
        //aquí se comprueba el entrenamiento. Lo compruebo cada npat veces.
      if((veces%npat)==0)
      {
        //contador=contador+1;
        errort=0.0;
        errort_ruido=0.0;
        k=0;
        for(p=0;p<npat;p++)
        {
            
            paso_red(w,u,capa,y,p);
        
            z[p]=y[0][ncapas-1][p];
            
            error_ruido[p]=pow((result[p]-z[p]),2);//error con respecto a los puntos con los que la red se entrena (q tienen ruido)
            error[p]=pow((sin(pi*y[0][0][p])-z[p]),2);//error de cada patrón con respecto al sen (función q qeremos aproximar)
            
            errort_ruido=errort_ruido+error_ruido[p];//Suma de errores con respecto a los datos con ruido.
            errort=errort+error[p];//Suma de errores de cada patrón
            
            if(error_ruido[p]<precision)//esto es por si durante el entrenamiento se alcanza la igualdad total con la función buscada, aunq para funciones
                k++;                //funciones q no sean muy sencillas será muy difícil q ocurra, y prácticamente imposible si hay ruido.
            
            if(k==npat)
                verdad=1;
            
        }
        
        err<<errort/npat<<endl;//el q graficamos es el referido al error de la función. MEAN SQUARE ERROR.
        e_ruido<<errort_ruido/npat<<endl;
      }
        
        /*if(errort<(precision*4))//nse si esta m parece mejor forma de evaluar el error q la otra. Si es vda q el error total vendrá bn pa ver evoluciones.
            verdad=1;//si lo vas a evaluar así entonces mejor incluyelo en el while
*/
    }
//SALIDA FINAL.  
    for(p=0;p<npat;p++)
    {
        //cout<<y[0][0][p]/*<<x[1][p]*/<<"   "<<z[p]<<"  "<<result[p]<<endl;
        grafica_ent<<y[0][0][p]<<"  "<</*z[p]<<"  "<<*/result[p]<<endl;
    }
    cout<<(ncapas-2)<<" capas ocultas con "<<m<<" neuronas cada una"<<endl;
    cout<<veces<<" nº de veces q se han modificado los pesos y umbrales"<<endl;
    cout<<errort/npat<<" Error promedio con respecto a la función a aproximar"<<endl;
    cout<<errort_ruido/npat<<" Error promedio puntos ruido"<<endl;
    cout<<npat<<" ejemplos de entrenamiento"<<endl;
    cout<<"  "<<endl;
    
    
    //probamos ahora la red con distintas entradas

    double y1[m][ncapas][n_test],z1[n_test];
    double error_test;
    
    y1[0][0][0]=lim_inferior;//la 1ª entrada es el ppio del intervalo, y a partir de ahí le vamos sumando el tamaño del intervalo dividido entre los n_test pasos.
    for(i=1;i<n_test;i++)
        y1[0][0][i]=y1[0][0][i-1]+((lim_superior-lim_inferior)/n_test);
    
    for(p=0;p<n_test;p++)
    {
        paso_red_test(w,u,capa,y1,p);
        z1[p]=y1[0][ncapas-1][p];
       
    }
    
    for(i=0;i<n_test;i++)
    {
        grafica_final<<y1[0][0][i]<<"  "<<z1[i]<</*"  "<<sin(pi*y1[0][0][i])<<*/endl;
        //cout<<x1[0][i]<</*x1[1][i]<<*//*"  "<<z1[i]<<"  "<<sin(x1[0][i])<<endl;
        error_test=error_test+pow((z1[i]-sin(pi*y1[0][0][i])),2);
    }
    
    //cout<<error_test/n_test<<" Error del test"<<endl;
    cout<<endl;
    //cout<<contador<<endl;*/
    return 0;
}

double sigmoide(double x, double b)
{
    double y;
    y=-1.0+2.0/(1.0+exp(-b*x));
    
    return y;
}

double derivada_sigm(double x, double b)
{
    double y;
    y=0.5*b*(1-x*x);
    return y;
}

void paso_red(double w[m][m][ncapas], double u[m][ncapas], int capa[ncapas], double y[m][ncapas][npat], int p)
{
    int i,j,k;
    double y0,b;
    b=10.0;
    
    for(k=0;k<(ncapas-1);k++)
    {
        for(j=0;j<capa[k+1];j++)
        {
            y0=u[j][k];
            for(i=0;i<capa[k];i++)
            {
                y0=y0+y[i][k][p]*w[j][i][k];
            }
            
            if(k==(ncapas-2))//si estamos en la úlyima capa la func. act. es la identidad. Sino es la sigmooide.
                y[j][k+1][p]=y0;
            else
                y[j][k+1][p]=sigmoide(y0,b);
            
        }
    }
    return;
}


void paso_red_test(double w[m][m][ncapas], double u[m][ncapas], int capa[ncapas], double y1[m][ncapas][n_test], int p)
{
    int i,j,k;
    double y0,b;
    b=10.0;
    
    for(k=0;k<(ncapas-1);k++)
    {
        for(j=0;j<capa[k+1];j++)
        {
            y0=u[j][k];
            for(i=0;i<capa[k];i++)
            {
                y0=y0+y1[i][k][p]*w[j][i][k];
            }
            
            if(k==(ncapas-2))//si estamos en la úlyima capa la func. act. es la identidad. Sino es la sigmooide.
                y1[j][k+1][p]=y0;
            else
                y1[j][k+1][p]=sigmoide(y0,b);
            
        }
    }
    return;
}



void vector_x(double a,double b, double x[n][npat])
{
    double div;
    int i;
    div=(b-a)/npat;
    x[0][0]=a;
    for(i=1;i<npat;i++)
        x[0][i]=x[0][i-1]+div;
    
    desordenar(x);
    
    return;
    
}

void desordenar(double x[n][npat])
{
    int q,k,i,j,g[npat];
    double aux[npat];
    bool completado,existe;
    
    for(i=0;i<npat;i++)
        aux[i]=x[0][i];// aux es el vector q recibe la funcion a partir de aquí
    
    q=7;
    dranini_(&q);
    
    
    g[0]=floor(dranu_()*npat);
    //q=g[0];
    x[0][0]=aux[g[0]];
    k=0;
    i=1;
    completado=false;
    
    while(completado==false)
    {
        g[i]=floor(dranu_()*npat);
        j=0;
        existe=false;
        while((j<i)&&(existe==false))
        {
            if(g[i]==g[j])
                existe=true;
            j=j+1;
        }
        if(existe==false)
        {
            k=k+1;
            x[0][k]=aux[g[i]];
            i=i+1;
        }
        
        if(k==(npat-1))
            completado=true;
        //i=i+1;
    }
    return;
}
		
		
		
		
		
		
