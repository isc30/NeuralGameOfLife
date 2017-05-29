class Neuron
{
    public output: number;
    public weights: number[];
    public deltas: number[];
    public bias: number;

    public constructor(inputCount: number)
    {
        this.bias = Math.random() * 2 - 1;

        this.weights = [];
        this.deltas = [];

        for (let i = 0; i < inputCount; i++)
        {
            this.weights.push(Math.random() * 2 - 1);
            this.deltas.push(0);
        }
    }

    public activate(value: number): void
    {
        this.output = 1 / (1 + Math.exp(-value));
    }
}

class NeuralLayer
{
    public neurons: Neuron[];
    public input: number[];

    public constructor(inputCount: number, neuronCount: number)
    {
        this.input = new Array<number>(inputCount);

        this.neurons = [];
        for (let i = 0; i < neuronCount; i++)
        {
            this.neurons.push(new Neuron(inputCount));
        }
    }

    public getNeuronCount(): number
    {
        return this.neurons.length;
    }

    public getInputCount(): number
    {
        return this.input.length;
    }

    public setInput(input: number[]): void
    {
        if (input.length !== this.getInputCount())
        {
            throw new Error("Invalid input length");
        }

        this.input = input;
    }

    public getOutput(): number[]
    {
        const output: number[] = [];

        for (const neuron of this.neurons)
        {
            output.push(neuron.output);
        }

        return output;
    }

    public calculate(): void
    {
        for (const neuron of this.neurons)
        {
            let value = neuron.bias;

            for (let w = 0; w < neuron.weights.length; w++)
            {
                value += neuron.weights[w] * this.input[w];
            }

            neuron.activate(value);
        }
    }
}

class NeuralNetwork
{
    public hiddenLayers: NeuralLayer[];
    public outputLayer: NeuralLayer;

    public getOutput(): number[]
    {
        return this.outputLayer.getOutput();
    }

    public getInputCount(): number
    {
        if (this.hiddenLayers.length > 0)
        {
            return this.hiddenLayers[0].getInputCount();
        }

        return this.outputLayer.getInputCount();
    }

    public getOutputCount(): number
    {
        return this.outputLayer.getNeuronCount();
    }

    public constructor(inputCount: number, outputCount: number, hiddenLayersNeuronsCount: number[] = [])
    {
        this.hiddenLayers = [];
        let currentInputCount = inputCount;

        for (const neuronCount of hiddenLayersNeuronsCount)
        {
            this.hiddenLayers.push(new NeuralLayer(currentInputCount, neuronCount));
            currentInputCount = neuronCount;
        }

        this.outputLayer = new NeuralLayer(currentInputCount, outputCount);
    }

    public propagate(input: number[]): number[]
    {
        if (input.length !== this.getInputCount())
        {
            throw new Error("Incorrect input length");
        }

        let layerInput: number[] = input;

        for (const layer of this.hiddenLayers)
        {
            layer.setInput(layerInput);
            layer.calculate();

            layerInput = layer.getOutput();
        }

        this.outputLayer.setInput(layerInput);
        this.outputLayer.calculate();

        return this.getOutput();
    }

    public train(input: number[], expectedOutput: number[], learn: number = 1, momentum: number = 0): number
    {
        if (input.length !== this.getInputCount())
        {
            throw new Error("Incorrect input length");
        }

        if (expectedOutput.length !== this.getOutputCount())
        {
            throw new Error("Incorrect output length");
        }

        this.propagate(input);

        let errorForPropagate: number = 0;
        let maxOutputError: number = 0;

        for (let n = 0; n < this.outputLayer.getNeuronCount(); n++)
        {
            const layer = this.outputLayer;
            const neuron = layer.neurons[n];
            const neuronOutput = neuron.output;
            const deltaError = expectedOutput[n] - neuronOutput;
            const neuronError = deltaError * this.derivative(neuronOutput);
            const neuronLearn = learn * neuronError;

            neuron.bias += neuronLearn * 1; // Optimize bias
            maxOutputError += deltaError; // return

            for (let i = 0; i < layer.getInputCount(); i++)
            {
                const previousDelta = neuron.deltas[i];
                const delta = neuronLearn * layer.input[i] + previousDelta * momentum;

                // Correct weights
                neuron.weights[i] += delta;
                neuron.deltas[i] = delta;

                // Recalculate error
                errorForPropagate += neuron.weights[i] * neuronError;
            }
        }

        for (let l = this.hiddenLayers.length - 1; l >= 0; l--)
        {
            const layer = this.hiddenLayers[l];
            let previousError: number = 0;

            for (let n = 0; n < layer.getNeuronCount(); n++)
            {
                const neuron = layer.neurons[n];
                const neuronOutput = neuron.output;
                const neuronError = errorForPropagate * this.derivative(neuronOutput);
                const neuronLearn = learn * neuronError;

                neuron.bias += neuronLearn * 1; // Optimize bias

                for (let i = 0; i < layer.getInputCount(); i++)
                {
                    const previousDelta = neuron.deltas[i];
                    const delta = neuronLearn * layer.input[i] + previousDelta * momentum;

                    // Correct weights
                    neuron.weights[i] += delta;
                    neuron.deltas[i] = delta;

                    // Recalculate error
                    previousError += neuron.weights[i] * neuronError;
                }
            }

            errorForPropagate = previousError;
            previousError = 0;
        }

        return maxOutputError;
    }

    protected derivative(value: number): number
    {
        return value * (1 - value);
    }
}

function conway(state: number[]): number
{
    const neighborCount = state[0] + state[1] + state[2] + state[3] + state[5] + state[6] + state[7] + state[8];

    if (state[4] === 0 && neighborCount === 3)
    {
        return 1;
    }

    if (state[4] === 1 && (neighborCount === 2 || neighborCount === 3))
    {
        return 1;
    }

    return 0;
}

const network = new NeuralNetwork(9, 1, [2]);
const trainings: Array<{input: number[], output: number[]}> = [];

// Generate all possible cases
for (let x = 0; x < Math.pow(2, 9); x++)
{
    // int to binary string
    let bin = (x >>> 0).toString(2);
    while (bin.length < 9) { bin = "0" + bin; }

    const table = bin.split("").map(v => parseInt(v));
    trainings.push({input: table, output: [conway(table)]});
}

// Train
let errorshow = 0;
let maxError = 0;
let iterations = 0;

do
{
    iterations++;
    errorshow++;

    maxError = 0;
    const errors: number[] = [];
    for (const training of trainings)
    {
        errors.push(network.train(training.input, training.output, 0.2, 0.1)); // 0.2, 0.1
    }
    maxError = Math.abs(errors[errors.map(e => Math.abs(e)).reduce((r, v, ci, a) => a[r] < v ? ci : r, 0)]);

    if (errorshow >= 1000)
    {
        console.log(`Max error: ${maxError} - Iterations: ${iterations}`);
        errorshow = 0;
    }
}
while (maxError > 0.09);

for (const training of trainings)
{
    console.log(`Input: [${training.input}], Expected: [${training.output}], Output: [${network.propagate(training.input)}]`);
}

console.log();
console.log(`Trained in ${iterations} iterations!`);
console.log();
