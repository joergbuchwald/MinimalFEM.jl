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

struct Triplets
    x::Array{Int}
    y::Array{Int}
    val::Array{Float64}
end

@enum type UX=1 UY=2 UXY=3

struct Constraint
    node::Int
    type::type
end


function readindata(constraintsfile, nodelistfile, elementlistfile, forcesfile)
    constraints = readdlm(constraintsfile, ' ', Int, '\n')
    nodelist = readdlm(nodelistfile, ' ', Float64, '\n')
    elements = readdlm(elementlistfile, ' ', Int, '\n')
    forces = readdlm(forcesfile, ' ', Float64, '\n')
    inp = inputdata(constraints, nodelist, elements, forces)
    return inp
end

function CalculateStiffnessMaitrix!(nodes, element, D, triplets)
    x = [nodes[element.nodeIds[1]+1,1],nodes[element.nodeIds[2]+1,1], nodes[element.nodeIds[3]+1,1]]
    y = [nodes[element.nodeIds[1]+1,2],nodes[element.nodeIds[2]+1,2], nodes[element.nodeIds[3]+1,2]]

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

    for i in 1:3
        for j in 1:3
            push!(triplets.x, 2 * element.nodeIds[i] + 1)
            push!(triplets.y, 2 * element.nodeIds[j] + 1)
            push!(triplets.val, K[2 * i - 1, 2 * j - 1])

            push!(triplets.x, 2 * element.nodeIds[i] + 1)
            push!(triplets.y, 2 * element.nodeIds[j] + 2)
            push!(triplets.val, K[2 * i - 1, 2 * j + 0])

            push!(triplets.x, 2 * element.nodeIds[i] + 2)
            push!(triplets.y, 2 * element.nodeIds[j] + 1)
            push!(triplets.val, K[2 * i + 0, 2 * j - 1])

            push!(triplets.x, 2 * element.nodeIds[i] + 2)
            push!(triplets.y, 2 * element.nodeIds[j] + 2)
            push!(triplets.val, K[2 * i + 0, 2 * j + 0])
        end


    end

end

function ApplyConstraints!(M, constraints)
    for i in 1:length(constraints)
        index = 2 * constraints[i].node
        index += 1
        if constraints[i].type == UX || constraints[i].type == UXY 
            M[:,index] .= 0
            M[index,:] .= 0
            M[index,index] = 1
        end
        index += 1
        if constraints[i].type == UY || constraints[i].type == UXY
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


elements = Vector{Element}(undef, size(input.elements)[1])


for i in 1:size(input.elements)[1]
    nodeIds = input.elements[i,:]
    elements[i] = Element(zeros(Float64,3,6),nodeIds)
end

defaultconstraint = Constraint(0,UXY)
constraints = fill(defaultconstraint,size(input.constraints)[1])

for i in 1:size(input.constraints)[1]
    node = input.constraints[i,1]
    constraint_type = type(input.constraints[i,2])
    constraints[i] = Constraint(node,constraint_type)
end

loads = zeros(Float64,2*size(input.nodes)[1])

for i in 1:size(input.forces)[1]
    node = Int(input.forces[i,1])
    loads[2*node+1] = input.forces[i,2]
    loads[2*node+2] = input.forces[i,3]
end

triplets = Triplets(Array{Int}(undef,0),Array{Int}(undef,0),Array{Float64}(undef,0))
nodes = input.nodes

for i in 1:size(input.elements)[1]
    CalculateStiffnessMaitrix!(nodes, elements[i], D, triplets)
end

globalK = sparse(triplets.x,triplets.y,triplets.val, 2*size(input.nodes)[1],2*size(input.nodes)[1])

ApplyConstraints!(globalK, constraints)

x = globalK\loads
