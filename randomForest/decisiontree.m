classdef decisiontree < handle
    properties
        x
        y
        idxs
        f_idxs
        used_splits
        min_leaf
        depth
        n_features
        m
        n
        lhs
        rhs
        var_idx
        val
        score
        isleaf
        split_col
    end
    methods
        function obj = fill_dt(obj,x, y, n_features, f_idxs, idxs, depth,min_leaf,used_splits)
            %Funcion de inicialización
            obj.x = x;
            obj.y = y;
            obj.isleaf=true;
            obj.idxs = idxs;
            obj.used_splits=used_splits;
            obj.min_leaf = min_leaf;
            obj.f_idxs =  f_idxs;
            obj.depth = depth;
            obj.n_features = length(f_idxs);
            obj.m= size(x,1);
            obj.n= size(x,2);
            obj.val = mean(y(idxs));
            obj.score = inf;
            obj.find_varsplit();
        end
        function obj = decisiontree()
            %Funcion de inicialización
            obj.x = [];
            obj.y = [];
            obj.used_splits=[];
            obj.isleaf=true;
            obj.idxs = [];
            obj.min_leaf = 0;
            obj.f_idxs =  [];
            obj.depth = 0;
            obj.n_features = 0;
            obj.m= 0;
            obj.n= 0;
            obj.val =0;
            obj.score = inf;
            %obj.find_varsplit();
        end
        function  find_varsplit(obj)
            %Función de búsqueda de la mejor variable sobre la que dividir el
            %dataset de entrada. Se desea que la división del dataset sea lo
            %más uniforme posible, por tanto, se utiliza la varianza como
            %métrica de la división.
            bestvar=inf; %parto de una varianza infinita y me voy quedando
            % con la menor
            if(obj.depth >0)
                for nfeature = 1:length(obj.f_idxs)
                    w1=length(find(obj.x(obj.idxs,obj.f_idxs(nfeature))==0));
                    w2=length(find(obj.x(obj.idxs,obj.f_idxs(nfeature))==1));
                    var1= var(obj.y(obj.x(obj.idxs,obj.f_idxs(nfeature))==0));
                    var2= var(obj.y(obj.x(obj.idxs,obj.f_idxs(nfeature))==1));
                    var_f=w1*var1+w2*var2;
                    %La varianza de una division es calculada como la suma
                    %ponderada de las varianzas de cada rama del split
                    if(var_f<bestvar) 
                        if(w1 >= obj.min_leaf) && (w2 >= obj.min_leaf)
                        %si la división obtiene casos con menos de cinco
                        %filas, no se tiene en cuenta
                            obj.isleaf=false;
                            obj.var_idx= obj.f_idxs(nfeature);
                            bestvar = var_f;  
%                         else
%                            w1;
%                            w2;
                        end
                    end
                end 
            end
            %   disp(obj.var_idx);
            obj.split_col = obj.x(obj.idxs,obj.var_idx);
            %valor de la columna sobre la que se realiza el split, en las
            %filas indicadas por los indices del bootstrap
            if(obj.isleaf == false)
                obj.used_splits = [obj.used_splits,  obj.var_idx];
                x_col  = obj.split_col;
                lhs = x_col == 0; %indices de la division izquierda
                rhs = find(x_col == 1); %indices de la division derecha
                lf_idxs = randperm(size(obj.x,2));
                [~,idx] = intersect(lf_idxs,obj.used_splits);
                lf_idxs(idx)=[];
                lf_idxs= lf_idxs(1:obj.n_features); %regresores a tener en cuenta en el nuevo nodo izquierdo
                rf_idxs = randperm(size(obj.x,2));
                [~,idx] = intersect(rf_idxs,obj.used_splits);
                rf_idxs(idx)=[];
                rf_idxs = rf_idxs(1:obj.n_features);%regresores a tener en cuenta en el nuevo nodo derecho
                obj.lhs = decisiontree().fill_dt(obj.x, obj.y, obj.n_features, lf_idxs, obj.idxs(lhs), obj.depth-1, obj.min_leaf,obj.used_splits);
                obj.rhs = decisiontree().fill_dt(obj.x, obj.y, obj.n_features, rf_idxs, obj.idxs(rhs), obj.depth-1, obj.min_leaf, obj.used_splits);
                %Se realiza la creación de dos nuevas divisones con la profundidad una unidad menor 
            end
            
                
        end
        function prediction = predict(obj,x)
            prediction = [];
            for i = 1:size(x,1)
                prediction=[prediction, obj.predict_row(x(i,:))];
            end
            prediction = prediction';
        end
        
        function prediction = predict_row(obj,xi)
            %La predicción se realiza fila a fila, llegando a los nodos 'hoja'
            %de cada decision tree
            if (obj.isleaf)
                if(length(obj.idxs)>1)
                    obj.idxs;
                end
                prediction = obj.val;
            else
                if(xi(1,obj.var_idx)==0)
                    t = obj.lhs;
                else if (xi(1,obj.var_idx)==1)
                    t = obj.rhs;
                    end
            end
                prediction = t.predict_row(xi);
            end
        end
        function oob = calc_oob(obj,x,y)
            %Esta función es llamada por el RF tras la creación de cada
            %decision tree con el objetivo de evaluar su capacidad de
            %predicción sobre los datos que quedaron fuera del bootstrap
            y_test = y;
            x_test = x;
            y_test(obj.idxs)=[];
            x_test(obj.idxs,:)=[];
            y_pred = obj.predict(x_test);
            oob=20*log10(norm(y_pred-y_test,2)/norm(y_test,2));
        end
        
    end
end
