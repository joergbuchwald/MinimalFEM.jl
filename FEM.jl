using DelimitedFiles
using LinearAlgebra
using SparseArrays

struct inputdata
    constraints::AbstractArray
    nodes::AbstractArray
    elements::AbstractArray
    forces::AbstractArray
end

struct Material
    ν::Float64
    E::Float64
end

struct Element
    B::AbstractArray
    nodeIds::Vector
end

struct Elements
    vector::Vector{Element}
end

@enum type UX=1 UY=2 UXY=3

struct Constraint
    node::Int
    type::type
end

struct Constraints
    vector::Vector{Constraint}
end

function readindata(constraintsfile, nodelistfile, elementlistfile, forcesfile)
    constraints = readdlm(constraintsfile, ' ', Int, '\n')
    nodelist = readdlm(nodelistfile, ' ', Float64, '\n')
    elements = readdlm(elementlistfile, ' ', Int, '\n')
    forces = readdlm(forcesfile, ' ', Float64, '\n')
    inp = inputdata(constraints, nodelist, elements, forces)
    return inp
end

function CalculateStiffnessMaitrix!(element, D, triplets)
    x = [input.nodes[element.nodeIds[1]+1,1],input.nodes[element.nodeIds[2]+1,1], input.nodes[element.nodeIds[3]+1,1]]
    y = [input.nodes[element.nodeIds[1]+1,2],input.nodes[element.nodeIds[2]+1,2], input.nodes[element.nodeIds[3]+1,2]]

    ones = [1, 1, 1]
    C = hcat(ones, x)
    C = hcat(C,y)

    IC = inv(C)

    for i in 1:3
        element.B[1, 2 * i - 1] = IC[2, i]
        element.B[1, 2 * i + 0] = 0.0
        element.B[2, 2 * i - 1] = 0.0
        element.B[2, 2 * i + 0] = IC[3, i]
        element.B[3, 2 * i - 1] = IC[3, i]
        element.B[3, 2 * i + 0] = IC[2, i]
    end

    K = transpose(element.B) * D * element.B * abs(det(C)) / 2.0

    trplt11 = (1, 1, 1.0)
    trplt12 = (1, 1, 1.0)
    trplt21 = (1, 1, 1.0)
    trplt22 = (1, 1, 1.0)

    for i in 1:3
        for j in 1:3
            trplt11 = (2 * element.nodeIds[i] + 1, 2 * element.nodeIds[j] + 1, K[2 * i - 1, 2 * j - 1])
            trplt12 = (2 * element.nodeIds[i] + 1, 2 * element.nodeIds[j] + 2, K[2 * i - 1, 2 * j + 0])
            trplt21 = (2 * element.nodeIds[i] + 2, 2 * element.nodeIds[j] + 1, K[2 * i + 0, 2 * j - 1])
            trplt22 = (2 * element.nodeIds[i] + 2, 2 * element.nodeIds[j] + 2, K[2 * i + 0, 2 * j + 0])
            push!(triplets,trplt11)
            push!(triplets,trplt12)
            push!(triplets,trplt21)
            push!(triplets,trplt22)
        end


    end

end

function ApplyConstraints!(M, constraints)
    for i in 1:length(constraints.vector)
        index = 2 * constraints.vector[i].node
        index += 1
        if constraints.vector[i].type == UX || constraints.vector[i].type == UXY 
            M[:,index] .= 0
            M[index,:] .= 0
            M[index,index] = 1
        end
        index += 1
        if constraints.vector[i].type == UY || constraints.vector[i].type == UXY
            M[:,index] .= 0
            M[index,:] .= 0
            M[index,index] = 1
        end
    end
end


input = readindata("constraints.inp", "nodelist.inp", "elementlist.inp", "forces.inp")

materialdata = Material(0.3, 2000)

D = [1.0 materialdata.ν 0.0; materialdata.ν 1.0 0.0; 0.0 0.0 (1-materialdata.ν)/2]

D *= materialdata.E / (1 - materialdata.ν^2)


elements = Elements(Vector{Element}(undef, size(input.elements)[1]))


for i in 1:size(input.elements)[1]
    nodeIds = input.elements[i,:]
    elements.vector[i] = Element(zeros(Float64,3,6),nodeIds)
end

defualtconstraint = defaultconstraint = Constraint(0,UXY)
constraints = Constraints(fill(defaultconstraint,size(input.constraints)[1]))

for i in 1:size(input.constraints)[1]
    node = input.constraints[i,1]
    constraint_type = type(input.constraints[i,2])
    constraints.vector[i] = Constraint(node,constraint_type)
end

loads = zeros(Float64,2*size(input.nodes)[1])

for i in 1:size(input.forces)[1]
    node = Int(input.forces[i,1])
    loads[2*node+1] = input.forces[i,2]
    loads[2*node+2] = input.forces[i,3]
end

triplets = Vector{Tuple{Int64,Int64,Float64}}(undef,0)

for i in 1:size(input.elements)[1]
    CalculateStiffnessMaitrix!(elements.vector[i], D, triplets)
end

X = Vector{Int}(undef,0)
Y = Vector{Int}(undef,0)
val = Vector{Float64}(undef,0)

for i in 1:length(triplets)
    push!(X,triplets[i][1])
    push!(Y,triplets[i][2])
    push!(val,triplets[i][3])
end

globalK = sparse(X,Y,val, 2*size(input.nodes)[1],2*size(input.nodes)[1])

ApplyConstraints!(globalK, constraints)

x = globalK\loads
