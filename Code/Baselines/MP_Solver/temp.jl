println("================패키지 부르기=================")
using JuMP, Cbc
using DataFrames
using CSV
#=

Toy example

=#
df = CSV.read("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/MP_Solver/new_nutrition_julia.csv")
df = select!(df, Not(1, 3, 4, 5))
df[!, [2]]

num_sample = 3000
gen_menu_list = []

for p in 1:num_sample
    println(p)

    ```
    A = [33.875  125.77  53.11 97.375 281.01;
    1.4548*4 59.65*4 6.844*4 10.8048*4 146.7496*4;
    16.72425*9 63.2502*9 30.2337*9 54.77175*9 88.8363*9]
    ```
    # Affine Matrix임 (선형 변환해주는 행렬) - Transpose 취해줘야 함
    A = [fill(1, length(df[:, 1])) df[:, 2] df[:, :Protein] df[:, 6] df[:, 10]+df[:, 11] df[:, 14] df[:, 12] df[:, 13] df[:, 7] df[:, 8] df[:, 9] df[:, 15] df[:, 16] df[:, 19] df[:, 20] df[:, 21] df[:, 22] df[:, 23] df[:, 24]]'

    # 변수의 lower_bound 제약 리스트 만들기
    # 갯수; 열량; 단백질; 식이섬유; 비타민A; 비타민C; 비타민B1 (티아민); 비타민B1 (리보플라빈); 칼슘; 철; 나트륨; 리놀레산; 알파리놀렌산; 간식; 밥국; 밥; 국; 반찬; 김치
    # l = [14; 945; 15; 8.25; 172.5; 26.25; 0.3; 0.375; 375; 3.75; 0; 3.3; 0.4; 2; 0; 2; 0; 2; 2]
    l = [14; 945; 15; 8.25; 172.5; 26.25; 0.3; 0.375; 375; 3.75; 0; 3.3; 0.4; 2; 0; 2; 2; 2; 2]

    # 변수의 upper_bound 제약 리스트
    # u = [14; 1155; Inf; 15; 562.5; 382.5; Inf; Inf; 1875; 30; 1200; 6.8; 0.9; 4; 2; 2; 2; 4; 2]
    u = [14; 1155; Inf; 15; 562.5; 382.5; Inf; Inf; 1875; 30; 1200; 6.8; 0.9; 4; 2; 2; 2; 4; 2]

    # 각종 인스턴스 정의
    m = Model(Cbc.Optimizer)            # 옵티마이저 모델의 인스턴스 정의 (COIN Branch-and-Cut Solver).
    index_x = 1:length(df[!, 3])        # 변수들의 인덱스 정의
    index_constraints_A = 1:19          # 제약조건들의 인덱스

    # 모델에 적용할 변수 정의 (각 변수의 하한은 0, 상한은 1로 세팅한다.)
    @variable(m, x[i in index_x], binary = true)

    # all_variables는 정의된 model에 존재하는 모든 변수를 list화하여 var라는 array에 변수화하여 담음.
    var = all_variables(m)  # 이 var는 m의 solver가 막 바꿔서 넣어보는 임의의 변수들의 리스트를 의미한다. 
    for i in var
        JuMP.set_binary(i)  # 각 i 변수가 {0, 1} 집합 내의 값을 갖도록하는 제약 추가. [0, 1]이 아님에 주의하라!
    end


    # 선형 혹은 비선형 문제를 풀도록 하는 목적함수 세운 후 model에 반영
    ## 모델 m을 활용하여 my_function을 maximization 하도록 objective function을 세워라 

    B = fill(1, length(df[:, 1]))
    if p > 1
        B[t] = 0
    end
    my_function = sum( B[i] * var[i] for i in index_x) 

    # my_function = sum( 1 * var[i] for i in index_x) 
    @objective(m, Max, my_function)

    # Model 하에서 각 j번째 제약에 대해서 모든 변수들의 합이 l[j]와 u[j]사이에 있도록 제약식을 세운 후 model에 반영
    # 예) j번째 제약: l[j] <= A[j, 1]*x1 + A[j, 1]*x2 + ... + A[j, 1]*x1726 <= u[j]
    # constraints[j] 에 j번째 제약
    @constraint(m, constraint[j in index_constraints_A],
                l[j] <= sum( A[j,i] * var[i] for i in index_x ) <= u[j] 
                )

    # @constraint(m, constraint_FAT_1, 
    #             0 <= sum( (-0.15*A[2,i]+9*A[5,i]) * var[i] for i in index_x ) <=Inf 
    #             )
    # @constraint(m, constraint_FAT_2, 
    #             0 <= sum( (0.30*A[2,i]-9*A[5,i]) * var[i] for i in index_x ) <=Inf
    #             )
    # @constraint(m, constraint_C_1, 
    #             0 <= sum( (-0.55*A[2,i]+4*A[3,i]) * var[i] for i in index_x ) <=Inf 
    #             )
    # @constraint(m, constraint_C_2, 
    #             -Inf <= sum( (-0.65*A[2,i]+4*A[3,i]) * var[i] for i in index_x ) <=0 
    #             )
    # @constraint(m, constraint_P_1, 
    #             0 <= sum( (-0.07*A[2,i]+4*A[4,i]) * var[i] for i in index_x ) <=Inf 
    #             )
    # @constraint(m, constraint_P_2, 
    #             -Inf <= sum( (-0.2*A[2,i]+4*A[4,i]) * var[i] for i in index_x ) <=0 
    #             )

    # 위에서 세운 objective function과 constraints를 반영하여 model을 최적화한다.
    optimize!(m)

    sol = Dict()
    for i in all_variables(m)
        # println(i)
        sol[i] = JuMP.value(i)
    end  

    gen_menu = [k for (k,v) in sol if v >= 0.9]
    # [gen_menu_list; gen_menu]
    append!(gen_menu_list, gen_menu)
    global t = parse(Int, chop(string(rand(gen_menu_list)), head=2))

    # vcat(gen_menu_list, gen_menu')
end

gen_menu_mat = reshape(gen_menu_list, (14, num_sample))
CSV.write("/home/messy92/Leo/Controlled_Sequence_Generation/Diet_Generation/Code/Baselines/MP_Solver/gen_menu_mat2.csv",  DataFrame(gen_menu_mat), header=false)
# for i in 1:14
#     println(i)
# end
# x = zeros(0)
# append!( x, rand(10) )

# v = rand(6)
# reshape(v, 3, 2)