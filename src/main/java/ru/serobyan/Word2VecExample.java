package ru.serobyan;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;

public class Word2VecExample {

    public static void main(String[] args) throws Exception {
        var file = Word2VecExample.class.getClassLoader().getResource("raw_sentences.txt").getFile();
        var iterator = new BasicLineIterator(file);
        var tokenizer = new DefaultTokenizerFactory();

        tokenizer.setTokenPreProcessor(new CommonPreprocessor());

        System.out.println("Building model....");
        var model = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iterator)
                .tokenizerFactory(tokenizer)
                .build();

        System.out.println("Fitting model....");
        model.fit();

        var wordsNearest = model.wordsNearest("day", 10);
        System.out.println("10 Words closest to 'day': " + wordsNearest); // 10 Words closest to 'day': [night, week, game, year, season, time, office, group, set, war]
    }
}