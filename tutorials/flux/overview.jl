# Overview of flux.jl
# https://fluxml.ai/Flux.jl/stable/models/overview/

# You can install flux using `Pkg.add("Flux")`
using Flux
using Flux: train!

# Let's try to predict a simple linear function
actual(x) = 6x - 11

# Generate some sample data
# We need matrices so let's use `hcat` which is typically used for horizontal addition of matrices
# but for 1-col matrices, it doesn't do anything.
x_train, x_test = hcat(0:5...), hcat(6:10...)
y_train, y_test = actual.(x_train), actual.(x_test)

# Create a 1-input 1-output model
model = Dense(1 => 1)
optimizer = Descent()           # A simple gradient descent optimizer

# Need to define a loss function
loss(x, y) = Flux.Losses.mse(model(x), y)

metrics() = println("""
    Model params: $(Flux.params(model))
    Train loss: $(loss(x_train, y_train))
    Test loss: $(loss(x_test, y_test))
""")

# Let's see how the untrained model does
metrics()

# Prep data
data = [(x_train, y_train)]

# Let's now train this baby
epochs = 200
println("Training model for $epochs epochs...")

for _ in 1:epochs
    # Perform training step
    train!(loss, Flux.params(model), data, optimizer)

    # Print metrics
    metrics()
    println()
end
