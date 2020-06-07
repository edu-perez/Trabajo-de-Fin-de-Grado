
// Archivo de cabecera que contiene las funciones relativas
// a nmeros aleatorios.

/*
	Las funciones externas en 'C' van con el guion bajo al final, y no entre
	medias como en el original fortran. V�se el curso de f�ica
	computacional de Pedro.
*/
double GenerarDoubleAleatorio(double inf, double sup);

// Inicializa el nmero aleatorio
extern "C" void dranini_(int *);

// Devuelve un aleatorio distribuido uniformemente en (0,1)
extern "C" double dranu_(void);

// Escribe en 'fort.entero' el estado completo del n aleatorio para continuar runs
extern "C" void dranwrite_(int *);

// Lee de 'fort.entero' el estado completo del n aleatorio almacenado para continuar runs
extern "C" void dranread_(int *);

// Devuelve entero aleatorio en (1,n)
extern "C" int idran_(int *);

// Devuelve un gaussiano de media 0 y varianza 1, obtenido con el algoritmo de "numerical inversion method"
extern "C" double drang_(void);

// Devuelve en x1 y x2 dos gaussianos de media 0 y varianza 1 obtenido con el m�odo de Box-Mller-Wiener
extern "C" double dranbm_(double *x1,double *x2);

// Idem que  dranwrite y read pero ademas recibe el nombre del fichero. Mejor que las anteriores.
extern "C" void dranwrite2_(int *,char *);
extern "C" void dranread2_(int *,char *);



//////////////////////////////////////////////////////////////////
//      FUNCTION:       GenerarDoubleAleatorio                  //
//      DESCRIPTION:    Generamos un nmero aleatorio entre los //
//                      limites dados.                          //
//                                                              //
//////////////////////////////////////////////////////////////////
double GenerarDoubleAleatorio(double inf, double sup){
        double res=0.0;

        res=dranu_(); // [0,1)
        res=(sup-inf)*res + inf;  // [inf,up)

        return res;
}








