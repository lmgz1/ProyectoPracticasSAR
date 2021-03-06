import json
from nltk.stem.snowball import SnowballStemmer
import os
import re


class SAR_Project:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de noticias
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm + ranking de resultado

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [("title", True), ("date", False),
              ("keywords", True), ("article", True),
              ("summary", True)]

    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.index = {}  # hash para el indice invertido de terminos --> clave: termino, valor: posting list.
        # Si se hace la implementacion multifield, se pude hacer un segundo nivel de hashing de tal forma que:
        # self.index['title'] seria el indice invertido del campo 'title'.
        self.sindex = {}  # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {}  # hash para el indice permuterm.
        self.docs = {}  # diccionario de documentos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {}  # hash de terminos para el pesado, ranking de resultados. puede no utilizarse
        self.news = {}  # hash de noticias --> clave entero (newid), valor: la info necesaria para diferenciar la noticia dentro de su fichero (doc_id y posición dentro del documento)
        self.tokenizer = re.compile("\W+")  # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish')  # stemmer en castellano
        self.show_all = False  # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False  # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False  # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################

    def set_showall(self, v):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v

    def set_snippet(self, v):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v

    def set_stemming(self, v):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    def set_ranking(self, v):
        """

        Cambia el modo de ranking por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON RANKING DE NOTICIAS

        si self.use_ranking es True las consultas se mostraran ordenadas, no aplicable a la opcion -C

        """
        self.use_ranking = v

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def index_dir(self, root, **args):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root" e indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """

        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        for dir, subdirs, files in os.walk(root):
            for filename in files:
                if filename.endswith('.json'):
                    fullname = os.path.join(dir, filename)
                    self.index_file(fullname)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################
        
        ####    STEEMING, BORJA   #####
        if self.stemming:    
            self.make_stemming()
        ####
        
        ####    PERMUTERM, JAVI ####
        if self.permuterm:
            self.make_permuterm()
        ####

    def index_file(self, filename):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Indexa el contenido de un fichero.

        Para tokenizar la noticia se debe llamar a "self.tokenize"

        Dependiendo del valor de "self.multifield" y "self.positional" se debe ampliar el indexado.
        En estos casos, se recomienda crear nuevos metodos para hacer mas sencilla la implementacion

        input: "filename" es el nombre de un fichero en formato JSON Arrays (https://www.w3schools.com/js/js_json_arrays.asp).
                Una vez parseado con json.load tendremos una lista de diccionarios, cada diccionario se corresponde a una noticia

        """

        doc_id = len(self.docs.keys())
        new_id = len(self.news.keys())
        self.docs[doc_id] = filename

        with open(filename) as fh:
            jlist = json.load(fh)
            for i, article in enumerate([new["article"] for new in jlist]):
                self.news[new_id] = (doc_id, i)
                for token in set(self.tokenize(article)):  # set() para eliminar repetidas
                    if token not in self.index:
                        self.index[token] = [new_id]
                    else:
                        self.index[token].append(new_id)
                new_id += 1
        
        
        #
        # "jlist" es una lista con tantos elementos como noticias hay en el fichero,
        # cada noticia es un diccionario con los campos:
        #      "title", "date", "keywords", "article", "summary"
        #
        # En la version basica solo se debe indexar el contenido "article"
        #
        #
        #
        
        #################
        ### COMPLETAR ###
        #################

    def tokenize(self, text):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()

    def make_stemming(self):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING.

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        self.stemmer.stem(token) devuelve el stem del token

        """
        
        ## Por cada token del índice...
        for token in self.index:
            
            ## Genero su stem.
            stemmedtoken = self.stemmer.stem(token)
            
            ## Si no tengo el stem en el índice de stems, lo añado creándo una lista con el token.
            if self.sindex.get(stemmedtoken) == None:
                self.sindex[stemmedtoken] = [token]
                
            ## Si tengo el stem en el índice de stems, significa que el stem del token es equivalente 
            ## al stem de otro token ya añadido, por lo tanto añado el token a la lista de tal stem.
            else :
                self.sindex[stemmedtoken].append(token)
        

        

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################
        
    def make_permuterm(self):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        """
        # Por cada token en el índice, añadimos el símbolo '$' como delimitador
        # Para cada longitud y rotación posible que contenga el símbolo '$',
        # # se crea una entrada y se añade el token a la lista
        for token in self.index.keys():
            pterm = token + '$'
            for i in range(len(pterm)):
                #for j in range(0, len(pterm) - 1):
                #print(i, j)
                #print(len(self.ptindex))
                    #item = pterm[j:]
                if '$' in pterm:
                    # Si ya existe en el índice permuterm, añadimos el token (si es nuevo) a su lista
                    if pterm in self.ptindex.keys():
                        if token not in self.ptindex.get(pterm):
                            self.ptindex[pterm].append(token)
                    # Y si no existía, se crea una entrada, con una lista de tokens
                    else:
                        self.ptindex[pterm] = [token]
                # Siguiente rotación del token
                pterm = pterm[1:] + pterm[0]


        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES
        
        Muestra estadisticas de los indices
        
        """
        print("=" * 40)
        if self.multifield:
            print("Number of indexed days: " + str(len(self.dates)))
            print("-" * 40)
            print("Number of indexed news: " + str(len(self.news)))
            print("-" * 40)
            print("TOKENS:")
            print("\t# tokens in 'title': " + str(len(self.title)))
            print("\t# tokens in 'date': " + str(len(self.dates)))
            print("\t# tokens in 'keywords': " + str(len(self.keywords)))
            print("\t# tokens in 'article': " + str(len(self.article)))
            print("\t# tokens in 'summary': " + str(len(self.summary)))
            print("-" * 40)
        else:
            #print("Number of indexed days: " + str(len(self.dates)))
            #print("-" * 40)
            print("Number of indexed news: " + str(len(self.news)))
            print("-" * 40)
            print("TOKENS: " + str(len(self.index)))
            print("-" * 40)
            #print("Positional queries are NOT allowed.")
            #print("-" * 40)

        if self.permuterm:

            if self.multifield:
                print("PERMUTERMS: ")
                print("\t# permuterms in 'title': " + str(len(self.pttitle)))
                print("\t# permuterms in 'date': " + str(len(self.ptdates)))
                print("\t# permuterms in 'keywords': " + str(len(self.ptkeywords)))
                print("\t# permuterms in 'article': " + str(len(self.ptarticle)))
                print("\t# permuterms in 'summary': " + str(len(self.ptsummary)))
                print("-" * 40)
            else:
                print("PERMUTERMS: " + str(len(self.ptindex)))
                print("-" * 40)

        if self.stemming == True:
            #if self.multifield == True:
                #print("STEMS:")
                #print("\t# stems in 'title': " + str(len(self.stitle)))
                #print("\t# stems in 'date': " + str(len(self.sdates)))
                #print("\t# stems in 'keywords': " + str(len(self.skeywords)))
                #print("\t# stems in 'article': " + str(len(self.sarticle)))
                #print("\t# stems in 'summary': " + str(len(self.ssummary)))
                #print("-" * 40)
            #else:
                print("STEMS: " + str(len(self.sindex)))
                print("-" * 40)
        
        print ("Parentheses queries are allowed")
        print("-" * 40)

        if self.positional == True:
            print("Positional queries are allowed.")
        else:
            print("Positional queries are NOT allowed.")
        print("=" * 40)

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def solve_query(self, query, prev={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        if query is None or len(query) == 0:
            return []

        if "(" in query:
            return self.solve_query_parenthesis(query)

        if len(query) == 1:
            if '?' in query:
                return self.get_permuterm(query)
            elif '*' in query:
                return self.get_permuterm(query)
            else:
                return self.get_posting(query)

        reg = re.compile(r"\w+")
        tokens = reg.findall(query)
        firstToken = tokens.pop(0)
        # Si el primer elemento no es un token, sino un conector 'NOT'
        if firstToken == 'NOT':
            connector = firstToken
            firstToken = tokens.pop(0)
            if '?' in firstToken:
                firstPosting = self.get_permuterm(firstToken)
            elif '*' in firstToken:
                firstPosting = self.get_permuterm(firstToken)
            else:
                firstPosting = self.get_posting(firstToken)
            firstPosting = self.reverse_posting(firstPosting)
        # Si el primer elemento es un token
        else:
            firstPosting = self.get_posting(firstToken)

        while len(tokens) > 1:
            connector = tokens.pop(0)
            nextToken = tokens.pop(0)

            # Si el siguiente elemento no es un token, sino un conector 'NOT'
            if nextToken == 'NOT':
                nextToken = tokens.pop(0)
                if '?' in nextToken:
                    nextPosting = self.get_permuterm(nextToken)
                elif '*' in nextToken:
                    nextPosting = self.get_permuterm(nextToken)
                else:
                    nextPosting = self.get_posting(nextToken)
                nextPosting = self.reverse_posting(nextPosting)
            # Si el siguiente elemento es un token
            else:
                nextPosting = self.get_posting(nextToken)

            # Según el conector de la solicitud
            if connector == 'AND':
                firstPosting = self.and_posting(firstPosting, nextPosting)
            if connector == 'OR':
                firstPosting = self.or_posting(firstPosting, nextPosting)
                
        if firstPosting is None:
            return []
        return firstPosting

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def solve_query_parenthesis(self, query, prev={}):
        """
        Resuelve una query con parentesis.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """
        tokenized = re.findall(r"(\(|\)|[\w|:]+)", query)

        return self._solve_query_parenthesis(tokenized)

    def _solve_query_parenthesis(self, query, ind=""):
        value = []
        conjunction = "OR"
        negation = False
        i = 0
        while i < len(query):
            token = query[i]
            if token == "(":
                n = 1
                i += 1
                n_query = []
                while True:
                    if query[i] == "(":
                        n += 1
                    elif query[i] == ")":
                        if n == 1:
                            break
                        n -= 1
                    n_query.append(query[i])
                    i += 1
                b = self._solve_query_parenthesis(n_query, ind+"    ")
                value = self.operate(value, b, conjunction, negation)
                negation = False
            elif token == "NOT":
                negation = True
            elif token == "AND" or token == "OR":
                conjunction = token
                #print(ind, token)
            else:
                value = self.operate(value, self.solve_query(token), conjunction, negation)
                negation = False
            i += 1

        return value.copy()


    def operate(self, a, b, op, not_b):
        if not_b:
            b = self.reverse_posting(b)
        if op == "AND":
            return self.and_posting(a, b)
        elif op == "OR":
            return self.or_posting(a, b)
        else:
            print("ERROR")


    def get_posting(self, term, field='article'):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve la posting list asociada a un termino. 
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        """
        
        #### STEMMING ####
        if self.use_stemming:
            return self.get_stemming(term)
        ####        ####
        
        
        return self.index.get(term, [])

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def get_positionals(self, terms, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        pass

        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################

    def get_stemming(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE STEMMING

        Devuelve la posting list asociada al stem de un termino.

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        # Generamos el stem del termino.
        stem = self.stemmer.stem(term)
        # Consultamos la lista de terminos pertenecientes a dicho stem.
        tokens = self.sindex.get(stem)
        
        # Stem no está en el índice?
        if tokens == None:
            return []
        
        r = aux = pl = []
        
        # Recorremos la lista de terminos.
        for token in tokens:
            
            # Obtenemos la posting list de cada termino con el mismo stem y las concatenamos
            pl = self.index.get(token)
            aux = self.or_posting(r, pl)
            r = aux
            
        # Eliminamos newid repetidos con la siguiente instrucción, al convertir a dict quitamos repetidos y
        # lo volvemos a convertir a lista
        #r = set(listapostings)
            
        # Ordenamos la lista
        r.sort()
            
        return r
               

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def get_permuterm(self, term, field='article'):
        """
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        
        term = term + '$'

        # Si contiene '?' rotamos hasta dejar el comodín al final de la palabra
        # Obtenemos la lista de palabras del diccionario de igual longitud, de las que obtendremos la unión de sus posting list
        if '?' in term:
            while term[-1] != '?':
                term = term[1:] + term[0]
            term = term[:-1]
            token_list = self.ptindex.get(term)
            
            # Extraemos los tokens con la longitud de la consulta
            for i in range(len(token_list)):
                if len(token_list[i]) != len(term):
                    del token_list[i]

            if len(token_list) == 1:
                return self.solve_query(token_list[0])

        # Si contiene '?' rotamos hasta dejar el comodín al final de la palabra
        # Obtenemos la lista de palabras del diccionario de igual longitud, de las que obtendremos la unión de sus posting list
        if '*' in term:
            while term[-1] != '*':
                term = term[1:] + term[0]
            term = term[:-1]
            token_list = []
            # Extraemos los tokens con la longitud mayor o igual a la consulta y que coincidan con su permuterm
            # Añadimos los tokens a la lista
            for clave in self.ptindex.keys():
                if clave[0:len(term)] == term:
                    for tk in self.ptindex.get(clave):
                        if tk not in token_list:
                            token_list.append(tk)

        if len(token_list) == 1:
            return self.solve_query(token_list[0])
        
        # Calculamos la consulta como union de tokens
        query = ''
        for i in range(len(token_list) - 1):
            query = query + token_list[i] + 'OR'
        query = query + token_list[len(token_list)]

        return self.solve_query(query)

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################

    def reverse_posting(self, p):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los newid exceptos los contenidos en p

        """
        r = []
        n = list(self.news.keys())
        
        #if p is None:
        # return n
        
        for new in n:
            if new not in p:
                r.append(new)

        return r

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def and_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular
        

        return: posting list con los newid incluidos en p1 y p2
        
        Estrategia :
            
        respuesta ← {}
        mientras No_FINAL( p1) AND No_FINAL( p2)
            hacer si docID (p1) = docID (p2)
                entonces Añadir (respuesta, docID (p1))
                p1 ← Avanzar_Siguiente(p1)
                p2 ← Avanzar_Siguiente(p2)
            sino si docID (p1) < docID (p2)
                entonces p1 ← Avanzar_Siguiente(p1)
            sino
                p2 ← Avanzar_Siguiente(p2)

        """

        r = []
        i = j = 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                r.append(p1[i])
                i = i + 1
                j = j + 1
            elif p1[i] < p2[j]:
                i = i + 1
            else:
                j = j + 1

        return r

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def or_posting(self, p1, p2):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 o p2
        
        Estrategia:
            

        respuesta ← {}
        mientras No_FINAL( p1) AND No_FINAL( p2)
            hacer si docID (p1) = docID (p2)
                entonces Añadir (respuesta, docID (p1))
                p1 ← Avanzar_Siguiente(p1)
                p2 ← Avanzar_Siguiente(p2)
            sino si docID (p1) < docID (p2)
                entonces Añadir (respuesta, docID (p1))
                p1 ← Avanzar_Siguiente(p1)
            sino 
            Añadir (respuesta, docID (p2))
                p2 ← Avanzar_Siguiente(p2)
        
        mientras No_FINAL( p1)
            hacer Añadir (respuesta, docID (p1))
            p1 ← Avanzar_Siguiente(p1)
                
        mientras No_FINAL( p2)
            hacer Añadir (respuesta, docID (p2))
            p2 ← Avanzar_Siguiente(p2)

        """

        r = []
        i = j = 0
        while i < len(p1) and j < len(p2):
            if (p1[i] == p2[j]):
                r.append(p1[i])
                i = i + 1
                j = j + 1
            elif p1[i] < p2[j]:
                r.append(p1[i])
                i = i + 1
            else:
                r.append(p2[j])
                j = j + 1

        while i < len(p1):  # Bucle que vacia la p1
            r.append(p1[i])
            i = i + 1

        while j < len(p2):  # Bucle que vacia la p2
            r.append(p2[j])
            j = j + 1

        return r

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se propone por si os es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los newid incluidos de p1 y no en p2

        """

        pass

        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################

    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados 

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T

        """
        result = self.solve_query(query)
        print("%s\t%d" % (query, len(result)))
        return len(result)  # para verificar los resultados (op: -T)

    def solve_and_show(self, query):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra informacion de las noticias recuperadas. Para ello a partir de la posting list proporcionada por solve_query extraemos
        la ruta al fichero JSON asociado a cada noticia correspondiente y extraemos la información necesaria
        Consideraciones:

        - En funcion del valor de "self.show_snippet" se mostrara una informacion u otra.
        - Si se implementa la opcion de ranking y en funcion del valor de self.use_ranking debera llamar a self.rank_result        

        param:  "query": query que se debe resolver.

        return: el numero de noticias recuperadas, para la opcion -T
        
        """
        result = self.solve_query(query)
        if self.use_ranking:
            result = self.rank_result(result, query)

        print('========================================')
        print('Query: '+str(query)+'\n')
        print('Number of results: '+str(len(result))+'\n')
        i=0
        
        for noticia in result:
            i=1+i
            fileId   = self.news[noticia]
            with open(self.docs[fileId[0]]) as f:
                jsonNoticia = json.load(f)            
            jsonNoticia=jsonNoticia[fileId[1]]
            print('#%s  (0) (%s) (%s) %s: (%s)  \n'%(i,noticia,jsonNoticia['date'],jsonNoticia['title'],jsonNoticia['keywords']))
            if(self.show_snippet):
                print('Summary: %s \n'%(jsonNoticia['summary']))
        print('========================================')
        return len(result)
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

    def rank_result(self, result, query):
        """
        NECESARIO PARA LA AMPLIACION DE RANKING

        Ordena los resultados de una query.

        param:  "result": lista de resultados sin ordenar , postinglist sin ordenar.
                "query": query, puede ser la query original, la query procesada o una lista de terminos. Query original en este caso.


        return: la lista de resultados ordenada

        """

        pass
        ###################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE RANKING ##
        ###################################################
