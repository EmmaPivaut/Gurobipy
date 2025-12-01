import sys
import gurobipy as gp
from gurobipy import GRB

def solve_videos(input_path):
    print(f"--- Lecture du fichier {input_path} ---")
    
    try:
        with open(input_path, 'r') as f:
            content = f.read().split()
    except FileNotFoundError:
        print(f"Erreur : Fichier {input_path} introuvable.")
        return
    
    iterator = iter(content)
    try:
        V = int(next(iterator))
        E = int(next(iterator))
        R = int(next(iterator))
        C = int(next(iterator))
        X = int(next(iterator))
        
        video_sizes = [int(next(iterator)) for _ in range(V)]
        
        endpoints = []
        for i in range(E):
            ld = int(next(iterator))
            k = int(next(iterator))
            connections = {}
            for _ in range(k):
                c_id = int(next(iterator))
                lc = int(next(iterator))
                connections[c_id] = lc
            endpoints.append({'ld': ld, 'conns': connections})
            
        requests = []
        for i in range(R):
            rv = int(next(iterator))
            re = int(next(iterator))
            rn = int(next(iterator))
            requests.append({'id': i, 'v': rv, 'e': re, 'n': rn})
            
    except StopIteration:
        print("Erreur de format.")
        return

    print(f"Données : {V} vidéos, {E} endpoints, {R} requêtes, {C} caches ({X}MB)")

  
    print("--- Construction du modèle ---")
    try:
        m = gp.Model("streaming_videos")
        m.setParam('OutputFlag', 1)
        m.setParam('MIPGap', 0.005) # 0.5%
        m.setParam('TimeLimit', 300)

        requested_videos = set(r['v'] for r in requests)
        
        # 2. Variables
        y = {}
        for c in range(C):
            for v in requested_videos:
                if video_sizes[v] <= X:
                    y[c, v] = m.addVar(vtype=GRB.BINARY, name=f"y[{c},{v}]")


        x = {}
        obj_terms = []
        
        for r in requests:
            v_id = r['v']
            e_id = r['e']
            count = r['n']
            ld = endpoints[e_id]['ld']
            
            for c_id, lc in endpoints[e_id]['conns'].items():
                
                if lc < ld and video_sizes[v_id] <= X:
                    saved = (ld - lc) * count
                    x[r['id'], c_id] = m.addVar(vtype=GRB.BINARY, name=f"x[{r['id']},{c_id}]")
                    obj_terms.append(saved * x[r['id'], c_id])

        # Définir l'objectif 
        m.setObjective(gp.quicksum(obj_terms), GRB.MAXIMIZE)

        # 3. Contraintes
        
        # C1: Capacité des caches
        for c in range(C):
         
            m.addConstr(
                gp.quicksum(video_sizes[v] * y[c, v] for v in requested_videos if (c, v) in y) <= X,
                name=f"Cap_C{c}"
            )

        # C2: 
        for (r_id, c_id), x_var in x.items():
            v_id = requests[r_id]['v']
            m.addConstr(x_var <= y[c_id, v_id])

        # C3: 
        for r in requests:
         
            vars_list = [x[r['id'], c] for c in endpoints[r['e']]['conns'] if (r['id'], c) in x]
            if vars_list:
                m.addConstr(gp.quicksum(vars_list) <= 1)

        # 4. Résolution
        print("Génération du fichier .mps...")
        m.write("videos.mps")
        print("Optimisation...")
        m.optimize()

        # 5. Export
        if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            print(f"Score final : {m.objVal:,.0f}")
            
            cache_contents = {}
            for (c, v), var in y.items():
                if var.X > 0.5:
                    if c not in cache_contents: cache_contents[c] = []
                    cache_contents[c].append(str(v))
            
            with open("videos.out", 'w') as f:
                f.write(f"{len(cache_contents)}\n")
                for c, vlist in cache_contents.items():
                    f.write(f"{c} {' '.join(vlist)}\n")
            print("Fichier videos.out généré.")
        else:
            print("Pas de solution trouvée.")

    except gp.GurobiError as e:
        print(f"Erreur Gurobi : {e}")

if __name__ == "__main__":
 
    if len(sys.argv) < 2:
        print("Usage: python videos.py [chemin_vers_dataset]")
        sys.exit(1)
    
    path = sys.argv[1]
    solve_videos(path)